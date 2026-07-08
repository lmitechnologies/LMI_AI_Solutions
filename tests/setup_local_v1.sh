#!/bin/bash
# Set up the v1 test env locally with uv (into .venv) — no anomalib v2.
# Mirrors the base + v1 stages of tests/dockerfile.tests, minus the container.
# Requires a local NVIDIA GPU + CUDA driver (the TRT engine tests need it).
#
# Unlike the container, tensorrt is installed as the pip wheel here (the nvcr
# base ships system TensorRT, so dockerfile.tests excludes it). A green local
# run is a fast iteration aid, not a full substitute for the container run.
#
# NOTE: `uv sync` makes .venv match the lock exactly, so re-running step 1 later
# REMOVES the git-installed detectron2/anomalib below. Re-run steps 2-4 after any
# `uv sync`.
set -e

# 1. GPU dependency group from the frozen lock + the project itself.
uv sync --frozen --group gpu

# 2. detectron2 — git-built against the torch just installed (not in the lock).
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2"

# 3. anomalib v1 stack (git-pinned; not in the lock). `anomalib install` shells out to
#    pip._internal, which a uv-managed .venv omits, so seed pip into the env first.
uv pip install jsonargparse==4.27.7 anomalib==1.1.1 pip
uv run anomalib install --option core

# 4. `anomalib install` shells out to pip, which perturbs the gpu group: it pulls numpy>=2
#    and leaves onnxruntime-gpu's package files half-clobbered. Re-assert both.
uv pip install 'numpy<2'
uv pip install --reinstall onnxruntime-gpu==1.21.0

echo "Done. Run tests with:  uv run bash tests/run_tests.sh all-v1"
