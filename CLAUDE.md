# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Working Rules

- **No assumptions.** Ask when unclear; don't guess.
- **Keep docstrings and comments short and concise.** Docstrings = public contract; comments = non-obvious *why*. Skip what the code already says; don't delete accurate ones.
- **Line width is 140** — applies to code, comments, and docstrings (Ruff E501, `pyproject.toml`).
- **TensorRT code must stay compatible with TensorRT 8.5.**

## Adding a Backend

A backend only registers if it's discoverable — see the `ModelRegistry` docstring in `lmi_common/model_registry.py` for the `PACKAGES` / `.model` / `@register` contract.

## Do Not Modify

`yolov5_lmi/`, `anomalib_lmi/v0/`, `legacy/`, `deprecated/` — do not touch unless explicitly instructed.

## Commits

Use Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) — prefixes drive the automated release (`.releaserc.json`).

## Gotcha

The two AD test suites need different, incompatible Anomalib versions — `test_v1.py` → `v1.1.1`, `test_v2.py` → `v2.*`. Don't install both in one env. See `tests/dockerfile.ci`.
