"""Training-config adapter for TolerantAnomalyDINO.

All model-specific dataset discovery, deterministic autosplitting, concise-YAML
expansion, capability selection, and artifact publication lives here.  The shared
``anomalib_lmi.v2.train`` entry point only calls generic optional model hooks.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Proven v8.4 defaults.  The user-facing YAML only needs normal model settings,
# target_precision, and data group names. Explicit model.params always win.
STANDARD_DEFAULTS: dict[str, Any] = {
    "num_neighbours": 1,
    "masking": False,
    "scoring_mode": "cosine",
    "pca_components": 32,
    # Bound the normal reference bank by default. This is independent of legacy
    # KCenterGreedy ratio mode and scales well to multi-million-patch datasets.
    "coreset_subsampling": False,
    "sampling_ratio": 0.1,
    "normal_bank_max_size": 65536,
    "normal_bank_cap_seed": 1337,
    "residual_creation_threshold": 0.10,
    "residual_topk_per_image": 20,
    "shared_residual_decontamination_enable": True,
    "shared_residual_decontamination_percentile": 95.0,
    "shared_residual_decontamination_use_magnitude": True,
    "shared_residual_decontamination_min_accept_samples": 8,
    "accept_damping": 0.0,
    "strict_stock_when_damping_one": True,
    "emit_patch_class": False,
    "image_accept_enable": True,
    "image_accept_threshold": 999.0,
    "image_accept_damping": 0.0,
    "image_reject_veto_enable": True,
    "image_reject_veto_threshold": 0.0,
    "image_accept_calibrate": True,
    "image_accept_max_reject_false_accept_rate": 0.05,
    "image_accept_safety_margin": 0.01,
    "image_accept_min_threshold": 0.0,
    "projected_image_accept_enable": True,
    "projected_image_accept_threshold": 999.0,
    "projected_image_accept_damping": 0.0,
    "projected_image_accept_calibrate": True,
    "projected_image_accept_max_reject_false_accept_rate": 0.05,
    "projected_image_accept_safety_margin": 0.01,
    "projected_image_accept_min_threshold": 0.0,
    "joint_image_accept_calibrate": True,
    "image_reject_boost_enable": True,
    "image_reject_boost_advantage_threshold": 0.0,
    "image_reject_boost_lambda": 0.0,
    "image_reject_boost_max": 0.75,
    "image_reject_boost_calibrate": True,
    "image_reject_boost_target_precision": 0.99,
    "residual_projection_enable": True,
    "residual_projection_dim": 32,
    "residual_projection_margin": 0.10,
    "residual_projection_epochs": 40,
    "residual_projection_steps_per_epoch": 64,
    "residual_projection_batch_size": 128,
    "residual_projection_lr": 2.0e-3,
    "residual_projection_weight_decay": 1.0e-4,
    "residual_projection_orthogonality_weight": 1.0e-3,
    "residual_projection_use_magnitude": True,
    "residual_projection_magnitude_transform": "log",
    "residual_projection_magnitude_scale": 1.0,
    "residual_projection_magnitude_clip": 5.0,
    "residual_projection_magnitude_eps": 1.0e-6,
    "residual_projection_seed": 1337,
    "residual_projection_fit_device": "cpu",
    "residual_projection_mode": "dual",
    "dual_projection_enable": True,
    "residual_projection_use_relative_xy": False,
    "residual_projection_xy_scale": 1.0,
    "residual_projection_intermediate_layers": (8,),
    "residual_projection_intermediate_scale": 1.0,
    "projected_reject_boost_enable": True,
    "projected_reject_boost_advantage_threshold": 0.0,
    "projected_reject_boost_lambda": 0.0,
    "projected_reject_boost_max": 0.75,
    "projected_reject_boost_calibrate": True,
    "projected_reject_boost_target_precision": 0.999,
    "projected_reject_boost_require_non_decreasing_overall_recall": True,
    "calibrate": True,
    "calibrate_t_normal_percentile": 95.0,
    "calibrate_t_known_percentile": 95.0,
    "calibration_normal_samples": 4096,
    "calibration_seed": 1337,
    "calibration_query_chunk_size": 256,
    "normal_bank_chunk_size": 65536,
    "residual_bank_chunk_size": 65536,
    "diagnostic_fail_on_overlap": True,
    "diagnostic_reject_class_depth": 1,
    "deployment_thresholds_enable": True,
    # Preserve both the image score and pixel anomaly map in their native raw domains.
    "return_raw_score": True,
    "return_raw_anomaly_map": True,
}

# Friendly TAD-only keys accepted in the standard-looking data block.  They are
# consumed here and never reach anomalib.data.Folder.
DATA_KEYS = {
    "acceptable_dir",
    "reject_dir",
    "defect_augmentations",
    "defect_augmentation_repeats",
    "defect_augmentation_include_original",
    "split_seed",
    "tad_auto_split",
    "postprocessor_val_ratio",
    "stratify_acceptable",
    "stratify_reject",
}

# If these are already supplied in model.params the caller is intentionally using
# the legacy/advanced explicit split interface; autosplitting stays out of the way.
EXPLICIT_PATH_KEYS = {
    "normal_reference_dir",
    "acceptable_dir",
    "reject_dir",
    "image_calibration_good_dir",
    "image_calibration_acceptable_dir",
    "image_calibration_reject_dir",
    "diagnostic_good_dir",
    "diagnostic_acceptable_dir",
    "diagnostic_reject_dir",
}


def _flatten_legacy_params(params: dict[str, Any]) -> dict[str, Any]:
    """Accept the older nested TAD YAML schema for backward compatibility."""
    params = dict(params)

    thresholds = params.pop("thresholds", None)
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            params.setdefault(key, value)

    calibrate = params.get("calibrate")
    if isinstance(calibrate, dict):
        params["calibrate"] = bool(calibrate.get("enable", True))
        if "t_normal_percentile" in calibrate:
            params.setdefault("calibrate_t_normal_percentile", calibrate["t_normal_percentile"])
        if "t_known_percentile" in calibrate:
            params.setdefault("calibrate_t_known_percentile", calibrate["t_known_percentile"])

    defect_data = params.pop("defect_data", None)
    if isinstance(defect_data, dict):
        params.setdefault("acceptable_dir", defect_data.get("acceptable_dir"))
        params.setdefault("reject_dir", defect_data.get("reject_dir"))
        if "batch_size" in defect_data:
            params.setdefault("defect_batch_size", defect_data["batch_size"])
        if "num_workers" in defect_data:
            params.setdefault("defect_num_workers", defect_data["num_workers"])

    return params


def _supported_init_params(model_class) -> set[str]:
    try:
        signature = inspect.signature(model_class.__init__)
    except (TypeError, ValueError):
        return set()
    return {name for name in signature.parameters if name != "self"}


def _expand_defaults(model_class, params: dict[str, Any]) -> dict[str, Any]:
    params = _flatten_legacy_params(params)
    target_precision = float(params.pop("target_precision", 0.999))
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("model.params.target_precision must be in (0, 1]")

    supported = _supported_init_params(model_class)
    merged: dict[str, Any] = {}
    for key, value in STANDARD_DEFAULTS.items():
        if not supported or key in supported:
            merged[key] = value
    merged.update(params)

    targets = tuple(sorted({0.95, 0.99, target_precision}))
    if not supported or "diagnostic_target_precisions" in supported:
        merged.setdefault("diagnostic_target_precisions", targets)
    if not supported or "deployment_target_precisions" in supported:
        merged.setdefault("deployment_target_precisions", targets)
    return merged


def _normalize_extensions(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_EXTENSIONS
    result: list[str] = []
    for ext in values:
        ext = str(ext).lower()
        if not ext.startswith("."):
            ext = "." + ext
        result.append(ext)
    return tuple(sorted(set(result)))


def _iter_images(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.as_posix(),
    )


def _stable_order(paths: Iterable[Path], *, root: Path, seed: int, salt: str) -> list[Path]:
    def key(path: Path) -> tuple[str, str]:
        rel = path.relative_to(root).as_posix()
        digest = hashlib.sha256(f"{seed}|{salt}|{rel}".encode("utf-8")).hexdigest()
        return digest, rel

    return sorted(paths, key=key)


def _split_one_stratum(
    paths: list[Path],
    *,
    root: Path,
    seed: int,
    salt: str,
    test_ratio: float,
    calibration_fraction_of_holdout: float,
) -> dict[str, list[Path]]:
    """Split one class/stratum deterministically into reference/calibration/test."""
    ordered = _stable_order(paths, root=root, seed=seed, salt=salt)
    n = len(ordered)
    if n == 0:
        return {"reference": [], "calibration": [], "test": []}
    if n == 1 or test_ratio <= 0.0:
        return {"reference": ordered, "calibration": [], "test": []}

    holdout = int(round(n * test_ratio))
    holdout = max(1, min(holdout, n - 1))

    if calibration_fraction_of_holdout <= 0.0:
        n_cal = 0
    elif holdout == 1:
        # Prefer calibration to a one-image final test set; deployment calibration
        # is more useful and final metrics on one image are not meaningful.
        n_cal = 1
    else:
        n_cal = int(round(holdout * calibration_fraction_of_holdout))
        n_cal = max(1, min(n_cal, holdout - 1))

    return {
        "reference": ordered[holdout:],
        "calibration": ordered[:n_cal],
        "test": ordered[n_cal:holdout],
    }


def _split_group(
    root: Path,
    paths: list[Path],
    *,
    group: str,
    seed: int,
    test_ratio: float,
    calibration_fraction_of_holdout: float,
    stratify: bool,
) -> dict[str, list[Path]]:
    if not stratify:
        return _split_one_stratum(
            paths,
            root=root,
            seed=seed,
            salt=group,
            test_ratio=test_ratio,
            calibration_fraction_of_holdout=calibration_fraction_of_holdout,
        )

    strata: dict[str, list[Path]] = {}
    for path in paths:
        rel = path.relative_to(root)
        stratum = rel.parts[0] if len(rel.parts) > 1 else "__root__"
        strata.setdefault(stratum, []).append(path)

    result = {"reference": [], "calibration": [], "test": []}
    for stratum in sorted(strata):
        part = _split_one_stratum(
            strata[stratum],
            root=root,
            seed=seed,
            salt=f"{group}:{stratum}",
            test_ratio=test_ratio,
            calibration_fraction_of_holdout=calibration_fraction_of_holdout,
        )
        for partition in result:
            result[partition].extend(part[partition])
    return result


def _safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def _materialize_split_tree(
    split_root: Path,
    source_roots: dict[str, Path],
    splits: dict[str, dict[str, list[Path]]],
) -> None:
    if split_root.exists():
        shutil.rmtree(split_root)
    for partition in ("reference", "calibration", "test"):
        for group, source_root in source_roots.items():
            group_out = split_root / partition / group
            group_out.mkdir(parents=True, exist_ok=True)
            for source in splits[group][partition]:
                rel = source.relative_to(source_root)
                _safe_symlink(source, group_out / rel)


def _path_or_none(path: Path, count: int) -> str | None:
    return str(path) if count > 0 else None


def _count(splits: dict[str, dict[str, list[Path]]], group: str, partition: str) -> int:
    return len(splits.get(group, {}).get(partition, []))


def _has_explicit_paths(params: dict[str, Any]) -> bool:
    return any(params.get(key) not in (None, "") for key in EXPLICIT_PATH_KEYS)


def _configure_defect_augmentations(params: dict[str, Any], data_cfg: dict[str, Any]) -> None:
    """Route Anomalib-style augmentation config into TAD tolerance-bank training.

    By default, when ``data.train_augmentations`` is present, ACCEPT/REJECT
    reference-bank construction inherits the same transform list. Users can
    override with ``data.defect_augmentations`` or disable synthetic tolerance
    views with ``data.defect_augmentation_repeats: 0``.
    """
    if "defect_augmentations" not in params:
        requested = data_cfg.get("defect_augmentations", "same_as_train")
        if isinstance(requested, str):
            mode = requested.strip().lower()
            if mode in {"same_as_train", "train", "inherit", "inherit_train"}:
                requested = copy.deepcopy(data_cfg.get("train_augmentations"))
            elif mode in {"none", "off", "disabled"}:
                requested = None
            else:
                raise ValueError(
                    "data.defect_augmentations must be a transform list, null, or 'same_as_train'"
                )
        params["defect_augmentations"] = copy.deepcopy(requested)

    augmentations_enabled = bool(params.get("defect_augmentations"))
    if "defect_augmentation_repeats" not in params:
        repeats = data_cfg.get("defect_augmentation_repeats", 1 if augmentations_enabled else 0)
        params["defect_augmentation_repeats"] = int(repeats)
    if "defect_augmentation_include_original" not in params:
        params["defect_augmentation_include_original"] = bool(
            data_cfg.get("defect_augmentation_include_original", True)
        )


def prepare_training_config(model_class, cfg: dict, *, config_path: Path) -> dict:
    """Resolve concise TAD YAML into a leakage-safe internal training config."""
    cfg = copy.deepcopy(cfg)
    model_cfg = cfg.setdefault("model", {})
    params = _expand_defaults(model_class, dict(model_cfg.get("params", {}) or {}))
    model_cfg["params"] = params
    data_cfg = cfg.setdefault("data", {})
    _configure_defect_augmentations(params, data_cfg)

    # Advanced/legacy explicit split paths remain supported and bypass autosplit.
    if _has_explicit_paths(params):
        logger.info("TAD explicit split paths detected; automatic splitting disabled.")
        for key in DATA_KEYS:
            data_cfg.pop(key, None)
        return cfg

    if not bool(data_cfg.get("tad_auto_split", True)):
        logger.info("TAD automatic splitting disabled by data.tad_auto_split=false.")
        for key in DATA_KEYS:
            data_cfg.pop(key, None)
        return cfg

    root = Path(data_cfg.get("root", "/app/data/")).expanduser()
    names = {
        "good": str(data_cfg.get("normal_dir", "train")),
        "acceptable": str(data_cfg.get("acceptable_dir", "acceptable")),
        "reject": str(data_cfg.get("reject_dir", "reject")),
    }
    source_roots = {group: root / name for group, name in names.items()}

    extensions = _normalize_extensions(data_cfg.get("extensions"))
    source_images = {group: _iter_images(path, extensions) for group, path in source_roots.items()}
    if not source_images["good"]:
        raise RuntimeError(
            f"TolerantAnomalyDINO requires normal images; none found in {source_roots['good']}"
        )

    seed = int(data_cfg.get("split_seed", data_cfg.get("seed", 1337)))
    test_mode = str(data_cfg.get("test_split_mode", "synthetic")).lower()
    test_ratio = 0.0 if test_mode == "none" else float(data_cfg.get("test_split_ratio", 0.2))
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("data.test_split_ratio must be in [0, 1)")

    val_mode = str(data_cfg.get("val_split_mode", "same_as_test")).lower()
    val_ratio = float(data_cfg.get("val_split_ratio", 0.5))
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("data.val_split_ratio must be in [0, 1]")
    calibration_fraction = 0.0 if val_mode == "none" else val_ratio

    splits = {
        "good": _split_group(
            source_roots["good"], source_images["good"], group="good", seed=seed,
            test_ratio=test_ratio, calibration_fraction_of_holdout=calibration_fraction,
            stratify=False,
        ),
        "acceptable": _split_group(
            source_roots["acceptable"], source_images["acceptable"], group="acceptable", seed=seed,
            test_ratio=test_ratio, calibration_fraction_of_holdout=calibration_fraction,
            stratify=bool(data_cfg.get("stratify_acceptable", True)),
        ),
        "reject": _split_group(
            source_roots["reject"], source_images["reject"], group="reject", seed=seed,
            test_ratio=test_ratio, calibration_fraction_of_holdout=calibration_fraction,
            stratify=bool(data_cfg.get("stratify_reject", True)),
        ),
    }

    out_root = Path(cfg["engine"]["default_root_dir"]).expanduser()
    workspace = out_root / ".tad_auto_split" / config_path.stem
    split_root = workspace / "splits"
    artifact_root = workspace / "artifacts"
    _materialize_split_tree(split_root, source_roots, splits)
    artifact_root.mkdir(parents=True, exist_ok=True)

    has_acceptable = bool(source_images["acceptable"])
    has_reject = bool(source_images["reject"])
    if has_acceptable and has_reject:
        tolerance_mode = "full"
    elif has_acceptable:
        tolerance_mode = "accept_only"
    elif has_reject:
        tolerance_mode = "reject_only"
    else:
        tolerance_mode = "normal_only"

    # Enable only capabilities supported by the available reference groups.
    if tolerance_mode == "full":
        params["acceptable_dir"] = str(split_root / "reference" / "acceptable")
        params["reject_dir"] = str(split_root / "reference" / "reject")
        logger.info("TAD data mode: full ACCEPT+REJECT tolerance model.")
    elif tolerance_mode == "accept_only":
        params.update({
            "acceptable_dir": str(split_root / "reference" / "acceptable"),
            "reject_dir": None,
            "image_accept_enable": True,
            "image_accept_threshold": 0.0,
            "projected_image_accept_enable": False,
            "image_reject_boost_enable": False,
            "projected_reject_boost_enable": False,
            "residual_projection_enable": False,
        })
        logger.warning(
            "TAD data mode: ACCEPT-only (%d acceptable, 0 reject). "
            "Standalone acceptable suppression enabled; reject precision/recall unavailable.",
            len(source_images["acceptable"]),
        )
    elif tolerance_mode == "reject_only":
        params.update({
            "acceptable_dir": None,
            "reject_dir": str(split_root / "reference" / "reject"),
            "image_accept_enable": False,
            "projected_image_accept_enable": False,
            "image_reject_boost_enable": True,
            "image_reject_boost_advantage_threshold": 0.0,
            "image_reject_boost_lambda": 1.0,
            "projected_reject_boost_enable": False,
            "residual_projection_enable": False,
        })
        logger.warning(
            "TAD data mode: REJECT-only (0 acceptable, %d reject). "
            "Standalone reject boosting and good-vs-reject threshold calibration enabled.",
            len(source_images["reject"]),
        )
    else:
        params.update({
            "acceptable_dir": None,
            "reject_dir": None,
            "image_accept_enable": False,
            "projected_image_accept_enable": False,
            "image_reject_boost_enable": False,
            "projected_reject_boost_enable": False,
            "residual_projection_enable": False,
        })
        logger.warning(
            "TAD data mode: normal-only. Base AnomalyDINO scoring with normal/FPR threshold fallback."
        )

    params["normal_reference_dir"] = str(split_root / "reference" / "good")

    for group, param_name in (
        ("good", "image_calibration_good_dir"),
        ("acceptable", "image_calibration_acceptable_dir"),
        ("reject", "image_calibration_reject_dir"),
    ):
        params[param_name] = _path_or_none(
            split_root / "calibration" / group,
            _count(splits, group, "calibration"),
        )

    params["image_calibration_output_dir"] = str(artifact_root)
    params["deployment_thresholds_output_dir"] = str(artifact_root)

    n_test_reject = _count(splits, "reject", "test")
    n_test_nonreject = _count(splits, "good", "test") + _count(splits, "acceptable", "test")
    diagnostics_available = n_test_reject > 0 and n_test_nonreject > 0
    params["diagnostic_enable"] = diagnostics_available
    params["diagnostic_good_dir"] = _path_or_none(
        split_root / "test" / "good", _count(splits, "good", "test")
    )
    params["diagnostic_acceptable_dir"] = _path_or_none(
        split_root / "test" / "acceptable", _count(splits, "acceptable", "test")
    )
    params["diagnostic_reject_dir"] = _path_or_none(
        split_root / "test" / "reject", _count(splits, "reject", "test")
    )
    params["diagnostic_output_dir"] = str(artifact_root)

    # Internal bookkeeping for the model-owned post-export publication hook.
    params["training_workspace"] = str(workspace)
    params["training_original_config_path"] = str(config_path)

    # The ordinary Folder datamodule is only used for the normal reference bank
    # and Anomalib postprocessor lifecycle. TAD-specific groups were consumed above.
    postprocessor_val_ratio = float(data_cfg.get("postprocessor_val_ratio", 0.05))
    if not 0.0 < postprocessor_val_ratio < 1.0:
        raise ValueError("data.postprocessor_val_ratio must be in (0, 1)")
    for key in DATA_KEYS:
        data_cfg.pop(key, None)
    data_cfg["root"] = str(split_root / "reference")
    data_cfg["normal_dir"] = "good"
    data_cfg["test_split_mode"] = "none"
    data_cfg["test_split_ratio"] = 0.0
    data_cfg["val_split_mode"] = "synthetic"
    data_cfg["val_split_ratio"] = postprocessor_val_ratio
    data_cfg.setdefault("seed", seed)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "mode": "tad_auto_split",
        "seed": seed,
        "input": {
            "root": str(root),
            "normal_dir": names["good"],
            "acceptable_dir": names["acceptable"],
            "reject_dir": names["reject"],
            "extensions": list(extensions),
        },
        "requested_split": {
            "test_split_mode": test_mode,
            "test_split_ratio": test_ratio,
            "val_split_mode": val_mode,
            "val_split_ratio": val_ratio,
        },
        "reference_augmentation": {
            "enabled": bool(params.get("defect_augmentations")),
            "repeats": int(params.get("defect_augmentation_repeats", 0)),
            "include_original": bool(params.get("defect_augmentation_include_original", True)),
            "inherits_train_augmentations": (
                data_cfg.get("defect_augmentations", "same_as_train") == "same_as_train"
                if isinstance(data_cfg.get("defect_augmentations", "same_as_train"), str)
                else False
            ),
        },
        "counts": {
            group: {
                "total": len(source_images[group]),
                **{
                    partition: len(splits[group][partition])
                    for partition in ("reference", "calibration", "test")
                },
            }
            for group in ("good", "acceptable", "reject")
        },
        "tolerance_mode": tolerance_mode,
        "capabilities": {
            "accept_suppression": tolerance_mode in {"accept_only", "full"},
            "reject_boost": tolerance_mode in {"reject_only", "full"},
            "comparative_projection": tolerance_mode == "full",
            "precision_calibration": has_reject,
        },
        "diagnostics_available": diagnostics_available,
        "partitions": {
            partition: {
                group: [str(path.resolve()) for path in splits[group][partition]]
                for group in ("good", "acceptable", "reject")
            }
            for partition in ("reference", "calibration", "test")
        },
    }

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (workspace / "resolved_train_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    logger.info(
        "TAD auto split: good=%s acceptable=%s reject=%s",
        manifest["counts"]["good"],
        manifest["counts"]["acceptable"],
        manifest["counts"]["reject"],
    )
    logger.info("TAD split manifest: %s", workspace / "split_manifest.json")
    return cfg


def _latest_version_dir(default_root: Path, model_name: str, data_name: str) -> Path | None:
    root = default_root / model_name / data_name
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("v") and item.name[1:].isdigit():
            candidates.append((int(item.name[1:]), item))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def publish_training_artifacts(model, cfg: dict) -> None:
    """Publish model-owned split/calibration artifacts beside exported weights."""
    workspace_value = getattr(model, "training_workspace", None)
    if not workspace_value:
        return
    workspace = Path(workspace_value)
    if not workspace.exists():
        logger.warning("TAD training workspace no longer exists: %s", workspace)
        return

    version_dir = _latest_version_dir(
        Path(cfg["engine"]["default_root_dir"]),
        str(cfg["model"]["class_name"]),
        str(cfg["data"].get("name", "dataset")),
    )
    if version_dir is None:
        logger.warning("Could not locate exported version directory for TAD artifacts.")
        return

    for name in ("split_manifest.json", "resolved_train_config.yaml"):
        source = workspace / name
        if source.exists():
            shutil.copy2(source, version_dir / name)

    original_value = getattr(model, "training_original_config_path", None)
    if original_value:
        original = Path(original_value)
        if original.exists():
            shutil.copy2(original, version_dir / "config.yaml")

    artifact_root = workspace / "artifacts"
    if artifact_root.exists():
        destination = version_dir / "tolerance_diagnostics"
        shutil.copytree(artifact_root, destination, dirs_exist_ok=True)
        for name in ("recommended_thresholds.json", "recommended_thresholds_raw.json"):
            source = artifact_root / name
            if source.exists():
                shutil.copy2(source, version_dir / name)

    logger.info("Published TAD split/calibration artifacts to %s", version_dir)
