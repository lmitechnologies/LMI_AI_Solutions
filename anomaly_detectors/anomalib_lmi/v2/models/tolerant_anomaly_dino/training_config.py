"""Training-config adapter for TolerantAnomalyDINO.

This module owns TAD-specific training convenience behavior:

* concise user-facing YAML -> internal reference/calibration/test paths
* deterministic, leakage-safe auto splitting
* ACCEPT-only / REJECT-only / full capability selection
* deployment/diagnostic target-precision expansion
* publishing TAD-owned artifacts beside exported weights

It intentionally does *not* hide model/scoring hyperparameters.  Parameters such
as ``residual_topk_per_image``, projection layers, reject-boost search grids,
projection fit device, and per-defect weights come from the model constructor or
from the YAML.  This keeps experimentally selected behavior visible and prevents
silent drift when constructor defaults change.
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

# Friendly TAD-only keys accepted in the ordinary Anomalib-style data block.
# They are consumed here and never reach anomalib.data.Folder.
DATA_KEYS = {
    "acceptable_dir",
    "reject_dir",
    "split_seed",
    "tad_auto_split",
    "postprocessor_val_ratio",
    "stratify_acceptable",
    "stratify_reject",
}

# Explicit path parameters mean the caller prepared the split manually and wants
# the advanced/legacy interface.  In that case autosplitting must stay out of the
# way, but target_precision is still translated below.
EXPLICIT_PATH_KEYS = {
    "normal_reference_dir",
    "image_calibration_good_dir",
    "image_calibration_acceptable_dir",
    "image_calibration_reject_dir",
    "diagnostic_good_dir",
    "diagnostic_acceptable_dir",
    "diagnostic_reject_dir",
}


def _supported_init_params(model_class) -> set[str]:
    """Return explicitly named constructor parameters for compatibility checks."""
    try:
        signature = inspect.signature(model_class.__init__)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _set_if_supported(
    params: dict[str, Any],
    supported: set[str],
    key: str,
    value: Any,
    *,
    overwrite: bool = True,
) -> None:
    """Set an internally generated constructor argument only when supported."""
    if supported and key not in supported:
        return
    if overwrite or key not in params:
        params[key] = value


def _flatten_legacy_params(params: dict[str, Any]) -> dict[str, Any]:
    """Accept the older nested TAD YAML schema without changing its semantics."""
    params = dict(params)

    thresholds = params.pop("thresholds", None)
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            params.setdefault(key, value)

    calibrate = params.get("calibrate")
    if isinstance(calibrate, dict):
        params["calibrate"] = bool(calibrate.get("enable", True))
        if "t_normal_percentile" in calibrate:
            params.setdefault(
                "calibrate_t_normal_percentile", calibrate["t_normal_percentile"]
            )
        if "t_known_percentile" in calibrate:
            params.setdefault(
                "calibrate_t_known_percentile", calibrate["t_known_percentile"]
            )

    defect_data = params.pop("defect_data", None)
    if isinstance(defect_data, dict):
        params.setdefault("acceptable_dir", defect_data.get("acceptable_dir"))
        params.setdefault("reject_dir", defect_data.get("reject_dir"))
        if "batch_size" in defect_data:
            params.setdefault("defect_batch_size", defect_data["batch_size"])
        if "num_workers" in defect_data:
            params.setdefault("defect_num_workers", defect_data["num_workers"])

    return params


def _prepare_user_model_params(
    model_class,
    params: dict[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    """Consume friendly meta-parameters while leaving model behavior explicit.

    ``target_precision`` is not a model-forward hyperparameter.  It describes the
    exported deployment operating points we want training to report.  Translate
    it into the model's diagnostic/deployment precision lists, then remove it so
    it can never leak into ``TolerantAnomalyDINO.__init__``.

    No scoring/projection/calibration hyperparameters are filled here.  Missing
    values use the model's real constructor defaults.
    """
    params = _flatten_legacy_params(params)
    supported = _supported_init_params(model_class)

    target_precision = params.pop("target_precision", None)
    if target_precision is not None:
        target_precision = float(target_precision)
        if not 0.0 < target_precision <= 1.0:
            raise ValueError("model.params.target_precision must be in (0, 1]")

        targets = tuple(sorted({0.95, 0.99, target_precision}))
        _set_if_supported(
            params,
            supported,
            "diagnostic_target_precisions",
            targets,
            overwrite=False,
        )
        _set_if_supported(
            params,
            supported,
            "deployment_target_precisions",
            targets,
            overwrite=False,
        )

    return params, supported


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


def _resolve_source(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _iter_images(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: path.as_posix(),
    )


def _stable_order(
    paths: Iterable[Path],
    *,
    root: Path,
    seed: int,
    salt: str,
) -> list[Path]:
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
    """Split one class/stratum deterministically into reference/calibration/test.

    With the standard Anomalib-style settings:

        test_split_ratio = 0.20
        val_split_ratio  = 0.50

    this gives approximately 80% reference / 10% calibration / 10% final test.
    """
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
        # Prefer calibration over a statistically meaningless one-image final test.
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

    # First directory below acceptable/ or reject/ is the diagnostic subtype.
    # Root-level images remain a valid stratum.
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


def _count(
    splits: dict[str, dict[str, list[Path]]],
    group: str,
    partition: str,
) -> int:
    return len(splits.get(group, {}).get(partition, []))


def _path_or_none(path: Path, count: int) -> str | None:
    return str(path) if count > 0 else None


def _has_explicit_split_paths(params: dict[str, Any]) -> bool:
    return any(params.get(key) not in (None, "") for key in EXPLICIT_PATH_KEYS)


def _apply_tolerance_mode(
    params: dict[str, Any],
    supported: set[str],
    *,
    tolerance_mode: str,
    split_root: Path,
) -> None:
    """Inject only availability-dependent capability changes.

    This function deliberately does not choose two-sided model hyperparameters.
    Full mode therefore uses exactly the YAML/constructor settings.  One-sided
    modes disable components that mathematically require the missing class and
    set only the natural one-sided decision boundary/fallback needed to function.
    """

    def setp(key: str, value: Any) -> None:
        _set_if_supported(params, supported, key, value, overwrite=True)

    if tolerance_mode == "full":
        setp("acceptable_dir", str(split_root / "reference" / "acceptable"))
        setp("reject_dir", str(split_root / "reference" / "reject"))
        logger.info("TAD data mode: full ACCEPT+REJECT tolerance model.")
        return

    if tolerance_mode == "accept_only":
        setp("acceptable_dir", str(split_root / "reference" / "acceptable"))
        setp("reject_dir", None)
        setp("image_accept_enable", True)
        # One-sided ACCEPT evidence is t_accept_known - d_accept, so zero is the
        # calibrated known-class boundary.
        setp("image_accept_threshold", 0.0)
        setp("image_reject_boost_enable", False)
        setp("projected_reject_boost_enable", False)
        setp("projected_image_accept_enable", False)
        setp("residual_projection_enable", False)
        logger.info("TAD data mode: ACCEPT-only; standalone suppression enabled.")
        return

    if tolerance_mode == "reject_only":
        setp("acceptable_dir", None)
        setp("reject_dir", str(split_root / "reference" / "reject"))
        setp("image_accept_enable", False)
        setp("projected_image_accept_enable", False)
        setp("image_reject_boost_enable", True)
        # One-sided REJECT evidence is d_reject - t_reject_known.  A0=0 is its
        # natural known-class boundary; lambda=1 is used only as a fallback if
        # held-out calibration cannot tune the boost.
        setp("image_reject_boost_advantage_threshold", 0.0)
        setp("image_reject_boost_lambda", 1.0)
        setp("projected_reject_boost_enable", False)
        setp("residual_projection_enable", False)
        logger.info("TAD data mode: REJECT-only; standalone boost enabled.")
        return

    if tolerance_mode != "normal_only":
        raise ValueError(f"Unknown TAD tolerance mode: {tolerance_mode}")

    setp("acceptable_dir", None)
    setp("reject_dir", None)
    setp("image_accept_enable", False)
    setp("projected_image_accept_enable", False)
    setp("image_reject_boost_enable", False)
    setp("projected_reject_boost_enable", False)
    setp("residual_projection_enable", False)
    logger.info("TAD data mode: normal-only; tolerance modifications disabled.")


def prepare_training_config(model_class, cfg: dict, *, config_path: Path) -> dict:
    """Resolve concise TAD YAML into the internal leakage-safe training config."""
    cfg = copy.deepcopy(cfg)
    model_cfg = cfg.setdefault("model", {})
    params, supported = _prepare_user_model_params(
        model_class, dict(model_cfg.get("params", {}) or {})
    )
    model_cfg["params"] = params

    data_cfg = cfg.setdefault("data", {})

    # Advanced explicit-split configurations remain fully supported.  Only the
    # friendly target_precision translation above is applied in this path.
    if _has_explicit_split_paths(params):
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
    normal_name = data_cfg.get("normal_dir", "train")
    acceptable_name = data_cfg.get("acceptable_dir", "acceptable")
    reject_name = data_cfg.get("reject_dir", "reject")

    source_roots = {
        "good": _resolve_source(root, normal_name),
        "acceptable": _resolve_source(root, acceptable_name),
        "reject": _resolve_source(root, reject_name),
    }

    extensions = _normalize_extensions(data_cfg.get("extensions"))
    source_images = {
        group: _iter_images(path, extensions) for group, path in source_roots.items()
    }
    if not source_images["good"]:
        raise RuntimeError(
            "TolerantAnomalyDINO requires normal images; "
            f"none found in {source_roots['good']}"
        )

    seed = int(data_cfg.get("split_seed", data_cfg.get("seed", 1337)))

    test_mode = str(data_cfg.get("test_split_mode", "synthetic")).lower()
    test_ratio = (
        0.0 if test_mode == "none" else float(data_cfg.get("test_split_ratio", 0.2))
    )
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("data.test_split_ratio must be in [0, 1)")

    val_mode = str(data_cfg.get("val_split_mode", "same_as_test")).lower()
    val_ratio = float(data_cfg.get("val_split_ratio", 0.5))
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("data.val_split_ratio must be in [0, 1]")
    calibration_fraction = 0.0 if val_mode == "none" else val_ratio

    splits = {
        "good": _split_group(
            source_roots["good"],
            source_images["good"],
            group="good",
            seed=seed,
            test_ratio=test_ratio,
            calibration_fraction_of_holdout=calibration_fraction,
            stratify=False,
        ),
        "acceptable": _split_group(
            source_roots["acceptable"],
            source_images["acceptable"],
            group="acceptable",
            seed=seed,
            test_ratio=test_ratio,
            calibration_fraction_of_holdout=calibration_fraction,
            stratify=bool(data_cfg.get("stratify_acceptable", True)),
        ),
        "reject": _split_group(
            source_roots["reject"],
            source_images["reject"],
            group="reject",
            seed=seed,
            test_ratio=test_ratio,
            calibration_fraction_of_holdout=calibration_fraction,
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

    _apply_tolerance_mode(
        params,
        supported,
        tolerance_mode=tolerance_mode,
        split_root=split_root,
    )

    _set_if_supported(
        params,
        supported,
        "normal_reference_dir",
        str(split_root / "reference" / "good"),
    )

    for group, param_name in (
        ("good", "image_calibration_good_dir"),
        ("acceptable", "image_calibration_acceptable_dir"),
        ("reject", "image_calibration_reject_dir"),
    ):
        _set_if_supported(
            params,
            supported,
            param_name,
            _path_or_none(
                split_root / "calibration" / group,
                _count(splits, group, "calibration"),
            ),
        )

    _set_if_supported(
        params,
        supported,
        "image_calibration_output_dir",
        str(artifact_root),
    )
    _set_if_supported(
        params,
        supported,
        "deployment_thresholds_output_dir",
        str(artifact_root),
    )

    n_test_reject = _count(splits, "reject", "test")
    n_test_nonreject = _count(splits, "good", "test") + _count(
        splits, "acceptable", "test"
    )
    diagnostics_available = n_test_reject > 0 and n_test_nonreject > 0

    _set_if_supported(
        params, supported, "diagnostic_enable", diagnostics_available
    )
    _set_if_supported(
        params,
        supported,
        "diagnostic_good_dir",
        _path_or_none(
            split_root / "test" / "good", _count(splits, "good", "test")
        ),
    )
    _set_if_supported(
        params,
        supported,
        "diagnostic_acceptable_dir",
        _path_or_none(
            split_root / "test" / "acceptable",
            _count(splits, "acceptable", "test"),
        ),
    )
    _set_if_supported(
        params,
        supported,
        "diagnostic_reject_dir",
        _path_or_none(
            split_root / "test" / "reject", _count(splits, "reject", "test")
        ),
    )
    _set_if_supported(
        params, supported, "diagnostic_output_dir", str(artifact_root)
    )

    # Optional bookkeeping fields supported by the model-owned artifact hook in
    # newer revisions. Older wrappers simply do not receive them.
    _set_if_supported(
        params, supported, "training_workspace", str(workspace)
    )
    _set_if_supported(
        params,
        supported,
        "training_original_config_path",
        str(config_path),
    )

    # Convert the friendly data block back into an ordinary Folder datamodule
    # containing only normal reference images. TAD calibration/test groups are
    # consumed through the model-owned paths above.
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
        "schema_version": 3,
        "mode": "tad_auto_split",
        "seed": seed,
        "input": {
            "root": str(root),
            "normal_dir": str(normal_name),
            "acceptable_dir": str(acceptable_name),
            "reject_dir": str(reject_name),
            "resolved_normal_dir": str(source_roots["good"]),
            "resolved_acceptable_dir": str(source_roots["acceptable"]),
            "resolved_reject_dir": str(source_roots["reject"]),
            "extensions": list(extensions),
        },
        "requested_split": {
            "test_split_mode": test_mode,
            "test_split_ratio": test_ratio,
            "val_split_mode": val_mode,
            "val_split_ratio": val_ratio,
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
        "partitions": {
            partition: {
                group: [
                    str(path.resolve()) for path in splits[group][partition]
                ]
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
        "TAD auto split: mode=%s good=%s acceptable=%s reject=%s",
        tolerance_mode,
        manifest["counts"]["good"],
        manifest["counts"]["acceptable"],
        manifest["counts"]["reject"],
    )
    logger.info("TAD split manifest: %s", workspace / "split_manifest.json")
    return cfg


def _latest_version_dir(
    default_root: Path,
    model_name: str,
    data_name: str,
) -> Path | None:
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
