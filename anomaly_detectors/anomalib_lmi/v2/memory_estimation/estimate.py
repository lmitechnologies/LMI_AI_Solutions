import logging

import torch

from anomaly_detectors.anomalib_lmi.v2.train import build_model, load_config

from . import make_memory_estimator
from .common import MemoryBudget, TileConfig
from .patchcore import PatchCoreMemoryEstimator

logger = logging.getLogger(__name__)

# VRAM headroom left free for the CUDA/cuDNN context, allocator fragmentation,
# framework overhead and other processes. This is memory the estimator's peak
# model does NOT account for: the CUDA context (~0.6-1 GiB), cuDNN convolution
# workspaces, backbone forward activations beyond the profiled feature peak, and
# non-PyTorch allocations. The observed unmodeled floor during PatchCore
# validation on an 8 GiB card is ~3.5 GiB. Overridable via `memory.reserve_mib`.
DEFAULT_RESERVE_MIB = 3_000
# Fallback device memory limit when CUDA is unavailable (e.g. CPU-only hosts).
FALLBACK_MEMORY_LIMIT_MIB = 24_000


def device_memory_limit_mib(device: str = "cuda") -> float:
    """Total VRAM (MiB) of the given CUDA device, or a fallback when unavailable."""
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        total_bytes = torch.cuda.get_device_properties(index).total_memory
        return total_bytes / 1024**2
    return FALLBACK_MEMORY_LIMIT_MIB


def estimate_max(train_config, num_samples=0, calibrate=None, reserve_mib=None):
    if isinstance(train_config, str):
        train_config = load_config(train_config)
    model_params = train_config["model"]["params"]
    precision = model_params.get("precision", "float32").lower()

    # Optional per-run memory tuning. `reserve_mib` is headroom the peak model does
    # not cover (CUDA context, cuDNN, fragmentation). `inference_chunk_size` controls
    # the runtime's nearest-neighbour chunk behaviour: unset/None auto-detects from
    # the installed anomalib, a positive int pins it, and 0 forces the conservative
    # un-chunked estimate (useful if the training runtime differs from this host).
    # An explicit `reserve_mib` argument overrides the config value.
    memory_cfg = train_config.get("memory") or {}
    if reserve_mib is None:
        reserve_mib = memory_cfg.get("reserve_mib", DEFAULT_RESERVE_MIB)
    inference_chunk_size = memory_cfg.get("inference_chunk_size")

    tile_size = model_params.get("tile_size")
    stride = model_params.get("stride")
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    tile_config = TileConfig(
        image_size=model_params["image_size"],
        tile_size=tile_size,
        stride=stride,
        batch_size=train_config["data"]["train_batch_size"],
    )

    # build_model mutates model_params (pops image_size/tile_size/stride), so read
    # everything we need from the config before building the model.
    coreset_sampling_ratio = model_params.get("coreset_sampling_ratio")

    estimator = make_memory_estimator(
        model=build_model(train_config["model"]),
        tile_config=tile_config,
        precision=precision,
        profiling_device="cuda",
    )

    profile = estimator.profile_features()

    if isinstance(calibrate, (list, tuple)):
        n_train, peak = calibrate
        factor = estimator.calibrate_workspace_factor(
            observed_peak_mib=peak,
            n_train_images=n_train,
            profile=profile,
            fixed_overhead_mib=0.0,
        )
        return round(factor, 4)

    memory_budget = MemoryBudget(
        memory_limit_mib=device_memory_limit_mib("cuda"),
        fixed_overhead_mib=0,
        safety_fraction=1.00,
        reserve_mib=reserve_mib,
    )

    # Only memory-bank estimators constrain the dataset size; each accepts a
    # different set of kwargs, so build them per model type.
    extra_kwargs = {}
    if isinstance(estimator, PatchCoreMemoryEstimator):
        extra_kwargs = {
            "coreset_sampling_ratio": coreset_sampling_ratio,
            "dataset_workspace_factor": 1.0515,
            # TODO: Add select factor based on image_size, (1.1641) based on 348x348
            "inference_chunk_size": inference_chunk_size,
        }

    estimate = estimator.estimate(
        memory_budget=memory_budget,
        n_train_images=num_samples,
        profile=profile,
        **extra_kwargs,
    )

    max_images = estimate.max_train_images

    # Report the peak for the dataset size actually used: the requested count when
    # given, otherwise the resolved cap (training reduces the dataset to it). At 0
    # images the peak is just activations and misrepresents the real run.
    peak_reference = num_samples or max_images
    if peak_reference and peak_reference != num_samples:
        del estimate
        estimate = estimator.estimate(
            memory_budget=memory_budget,
            n_train_images=peak_reference,
            profile=profile,
            **extra_kwargs,
        )

    peak_mib = estimated_peak_mib(estimate)
    del estimate
    return max_images, peak_mib


def estimated_peak_mib(estimate):
    """Estimated peak training memory (MiB), independent of model family.

    For memory-bank models with a requested image count this is the modelled peak
    at that count; otherwise it falls back to the fixed footprint (activations +
    params/optimizer/stats), which is the meaningful figure for gradient-trained
    models where memory does not scale with dataset size.
    """
    requested_peak = estimate.requested.get("requested_total_peak_mib")
    if requested_peak is not None:
        return requested_peak

    budget = estimate.budget
    return (
        budget.get("fixed_overhead_mib", 0.0)
        + budget.get("activation_peak_mib", 0.0)
        + budget.get("params_total_mib", 0.0)
        + budget.get("padim_stats_peak_mib", 0.0)
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--config", type=str, help="Config yaml path")
    ap.add_argument("-n", "--num_samples", type=int)
    ap.add_argument("-c", "--calibrate", type=int, nargs=2, help="(N, peak_mib)")
    ap.add_argument(
        "-r",
        "--reserve_mib",
        type=int,
        default=None,
        help="VRAM headroom (MiB) left free for CUDA/cuDNN context and allocator overhead; overrides the config's memory.reserve_mib.",
    )
    args = ap.parse_args()

    out = estimate_max(
        args.config,
        args.num_samples,
        calibrate=args.calibrate,
        reserve_mib=args.reserve_mib,
    )

    # Calibration returns a single factor; estimation returns (max_images, peak_mib).
    if isinstance(out, tuple):
        max_images, peak_mib = out
        print(f"max_train_images={max_images}")
        print(f"estimated_peak_mib={peak_mib:.2f}")
    else:
        print(out)
