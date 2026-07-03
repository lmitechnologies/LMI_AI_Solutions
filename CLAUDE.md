# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Working Rules

- **No assumptions.** Ask when unclear; don't guess.
- **Keep docstrings and comments short and concise.** Docstrings = public contract; comments = non-obvious *why*. Skip what the code already says; don't delete accurate ones. This applies to this file — keep it terse.
- **Line width is 140** — applies to code, comments, and docstrings (Ruff E501, `pyproject.toml`).
- **TensorRT code must stay compatible with TensorRT 8.5.**

## Adding a Backend

A backend only registers if it's discoverable — see the `ModelRegistry` docstring in `lmi_common/model_registry.py` for the `PACKAGES` / `.model` / `@register` contract.

## Do Not Modify

`yolov5_lmi/`, `anomalib_lmi/v0/`, `legacy/`, `deprecated/` — do not touch unless explicitly instructed.

## Running Tests Locally

Needs NVIDIA Container Toolkit + a GPU. From the repo root:

```bash
docker compose -f tests/docker-compose.yaml up --build
```

See `tests/docker-compose.yaml` and `tests/dockerfile.tests` for the services and images. The `v1`/`v2` split isolates the incompatible Anomalib versions (`test_v1.py` → `1.1.1`, `test_v2.py` → `2.*`, which can't share an env). Each service runs `bash tests/run_tests.sh <arg>`; HTML reports land in `tests/outputs/`.

## Commits

Use Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) — prefixes drive the automated release (`.releaserc.json`).
