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
# REMOVES the git-installed detectron2 below. Re-run step 2 after any `uv sync`.
set -e

# 1. GPU + anomalib v1 dependency groups from the frozen lock, plus the project itself.
uv sync --frozen --group gpu --group ad-v1

# 2. detectron2 — git-built against the torch just installed (not in the lock).
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2"

echo "Done. Run tests with:  uv run bash tests/run_tests.sh all-v1"
