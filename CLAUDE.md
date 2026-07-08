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

Needs a GPU. Run tests yourself; prefer a targeted run while iterating. HTML reports land in `tests/outputs/`.

Fastest, containerless path for the v1 suite on a GPU host: `bash tests/setup_local_v1.sh` (builds a uv `.venv`), then
`uv run bash tests/run_tests.sh all-v1`. Iteration aid only — containers remain the source of truth for CI parity, and v2 needs the container.

In containers (services + suite groups live in `tests/docker-compose.yaml` and `tests/run_tests.sh`):

```bash
# a suite group (all-v1 | od | utils | cls | ad-v1), building any missing TRT engines first
docker compose -f tests/docker-compose.yaml run --rm test_ais_v1_all bash tests/run_tests.sh od
# full v1+v2 suite — final verification
docker compose -f tests/docker-compose.yaml up --build
```

The `v1`/`v2` split isolates incompatible Anomalib versions (`test_v1.py` → `1.1.1`, `test_v2.py` → `2.*`) that can't share an env, so
anomalib v2 uses its own service and needs `--no-deps` (else `run` pulls in the whole v1 suite):

```bash
docker compose -f tests/docker-compose.yaml run --rm --no-deps test_ais_ad_v2 bash tests/run_tests.sh ad-v2
```

## Commits

Use Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) — prefixes drive the automated release (`.releaserc.json`).
