"""

Determines whether the current push/PR only touched documentation or
non-code files, and writes `skip_tests=true|false` to $GITHUB_OUTPUT.

Full git history is available (fetch-depth: 0 in actions/checkout).

Environment variables (injected by the workflow step):
  EVENT_NAME     github.event_name  ("push" | "pull_request")
  BASE_REF       github.base_ref    (PR target branch, e.g. "main")
  BEFORE         github.event.before (push: SHA of commit before the push)
  GITHUB_OUTPUT  path to the GitHub Actions output file
"""

import os
import re
import subprocess

# ---------------------------------------------------------------------------
# Configuration — edit these to change what counts as "docs only"
# ---------------------------------------------------------------------------

# Files matching any of these patterns are considered non-code and will NOT
# trigger the test suite on their own.
DOCS_PATTERNS: list[re.Pattern] = [
    re.compile(r"^docs/"),
    re.compile(r"\.md$"),
    re.compile(r"^\.gitignore$"),
    re.compile(r"^\.gitattributes$"),
    re.compile(r"^\.pre-commit-config\.yaml$"),
]

# Files whose change requires rebuilding the CI test Docker images
REQ_PATTERNS: list[re.Pattern] = [
    re.compile(r"^tests/requirements-base\.txt$"),
    re.compile(r"^tests/requirements-ci\.txt$"),
    re.compile(r"^tests/dockerfile\.ci$"),
]

# Files under .github/ are also skipped by default …
GITHUB_DIR_PATTERNS: list[re.Pattern] = [re.compile(r"^\.github/")]

# … unless they match this pattern, in which case they DO trigger tests.
CI_WORKFLOW_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\.github/workflows/ci\.yaml$"),
    re.compile(r"^\.github/workflows/scripts/check_changes\.py$"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def resolve_base(event_name: str, base_ref: str, before: str) -> str:
    null_sha = "0" * 40

    if event_name == "pull_request":
        run(["git", "fetch", "origin", base_ref])
        return f"origin/{base_ref}"

    if not before or before == null_sha:
        print("ℹ️  Initial push detected — using HEAD~1 as base.", flush=True)
        return "HEAD~1"

    return before


def get_changed_files(base: str) -> list[str]:
    result = subprocess.run(["git", "diff", "--name-only", base, "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  Could not diff against {base} (force push may have orphaned the commit) — falling back to HEAD~1.", flush=True)
        result = run(["git", "diff", "--name-only", "HEAD~1", "HEAD"])

    return [line for line in result.stdout.splitlines() if line.strip()]


def is_code_file(path: str) -> bool:
    """Return True if the file should trigger the test suite."""
    if any(p.search(path) for p in DOCS_PATTERNS):
        return False

    if any(p.match(path) for p in GITHUB_DIR_PATTERNS):
        return any(p.match(path) for p in CI_WORKFLOW_PATTERNS)

    return True


def write_output(key: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        # Fallback for local testing
        print(f"[GITHUB_OUTPUT] {key}={value}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    event_name = os.environ.get("EVENT_NAME", "push")
    base_ref = os.environ.get("BASE_REF", "")
    before = os.environ.get("BEFORE", "")

    base = resolve_base(event_name, base_ref, before)
    print(f"📊 Comparing against: {base}", flush=True)

    changed = get_changed_files(base)

    if not changed:
        print("ℹ️  No files changed (identical commits) — skipping tests.", flush=True)
        write_output("skip_tests", "true")
        return

    print("📝 Changed files:", flush=True)
    for f in changed:
        print(f"   {f}", flush=True)

    code_files = [f for f in changed if is_code_file(f)]

    req_files = [f for f in changed if any(p.match(f) for p in REQ_PATTERNS)]
    if req_files:
        print("\n🐳 CI image files changed — Docker rebuild required:", flush=True)
        for f in req_files:
            print(f"   {f}", flush=True)
    write_output("req_changed", "true" if req_files else "false")

    if not code_files:
        print("\n✅ Only docs/config changed — skipping tests.", flush=True)
        write_output("skip_tests", "true")
    else:
        print("\n⚠️  Code changes detected — running full test suite.", flush=True)
        print("Files requiring tests:", flush=True)
        for f in code_files:
            print(f"   {f}", flush=True)
        write_output("skip_tests", "false")


if __name__ == "__main__":
    main()
