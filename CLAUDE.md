# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Working Rules

- **No assumptions.** Ask when unclear; don't guess.
- **Keep docstrings and comments short and concise.** Docstrings = public contract; comments = non-obvious *why*. Skip what the code already says; don't delete accurate ones. This applies to this file — keep it terse.
- **Line width is 140** — applies to code, comments, and docstrings (Ruff E501, `pyproject.toml`).
- **TensorRT code must stay compatible with TensorRT 8.5.**

## Adding a Backend

Backends are declared in each registry's `BACKENDS` table — see the `ModelRegistry` docstring in `lmi_common/model_registry.py`.
Add one entry with the metadata lists and a `"module:ClassName"` `class_path`. Keep registry modules free of backend imports so
duplicate-key validation (run at registry import) works in any environment.

## Do Not Modify

`yolov5_lmi/`, `anomalib_lmi/v0/`, `legacy/`, `deprecated/` — do not touch unless explicitly instructed.

## Running Tests Locally

Needs NVIDIA Container Toolkit + a GPU. Run tests yourself, directly in these containers. Prefer a targeted run over the full suite
while iterating:

```bash
# plain pytest on a file/dir (no TRT engine building)
docker compose -f tests/docker-compose.yaml run --rm test_ais_v1_all pytest tests/lmi_common/ -q

# a suite group, building any missing TRT engines first: all-v1 | od | utils | cls | ad-v1 | ad-v2
docker compose -f tests/docker-compose.yaml run --rm test_ais_v1_all bash tests/run_tests.sh od

# full suite (v1 then v2) — for final verification
docker compose -f tests/docker-compose.yaml up --build
```

The `v1`/`v2` split isolates the incompatible Anomalib versions (`test_v1.py` → `1.1.1`, `test_v2.py` → `2.*`, which can't share an
env). Anything Anomalib v2 must use the v2 service with `--no-deps` (otherwise `run` triggers the full v1 suite it depends on):

```bash
docker compose -f tests/docker-compose.yaml run --rm --no-deps test_ais_ad_v2 bash tests/run_tests.sh ad-v2
```

See `tests/docker-compose.yaml` and `tests/dockerfile.tests` for the services and images. HTML reports land in `tests/outputs/`.

## Commits

Use Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) — prefixes drive the automated release (`.releaserc.json`).
