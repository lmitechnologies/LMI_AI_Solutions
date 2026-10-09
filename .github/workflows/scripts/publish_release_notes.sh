#!/usr/bin/env bash
# Puts the archived release notes above the generated commit list in the GitHub release.
# Usage: publish_release_notes.sh <git tag>
# Required env vars: GH_TOKEN, GITHUB_REPOSITORY
set -euo pipefail

tag="$1"
notes="docs/releases/${tag}.md"

if [[ ! -f "$notes" ]]; then
  echo "No $notes; leaving the release body as is."
  exit 0
fi

body="$(mktemp)"
{
  cat "$notes"
  printf '\n---\n\n'
  gh release view "$tag" --repo "$GITHUB_REPOSITORY" --json body -q .body
} > "$body"
gh release edit "$tag" --repo "$GITHUB_REPOSITORY" --notes-file "$body"
