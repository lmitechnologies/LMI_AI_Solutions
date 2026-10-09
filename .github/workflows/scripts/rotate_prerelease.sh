#!/usr/bin/env bash
# Archives PRERELEASE.md as docs/releases/<tag>.md and resets it to an empty template.
# Usage: rotate_prerelease.sh <git tag>
# Required env vars: GITHUB_REPOSITORY
set -euo pipefail

tag="$1"
src=PRERELEASE.md
dest="docs/releases/${tag}.md"
repo_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:?}"

# The empty template has no sections.
if grep -q '^## ' "$src"; then
  mkdir -p docs/releases
  # Relative PR links only resolve from the repo root.
  sed -E -e "1s/^# Pre-Release Notes — (.*) → next$/# Release Notes — \1 → ${tag}/" -e "s#\]\(\.\./\.\./pull/([0-9]+)\)#](${repo_url}/pull/\1)#g" "$src" > "$dest"
  echo "Archived $src to $dest."
else
  echo "No notes in $src; nothing to archive."
fi

cat > "$src" <<EOF
# Pre-Release Notes — ${tag} → next

Breaking changes and new features since \`${tag}\`. Each breaking change says what to update.
EOF
