#!/usr/bin/env bash
# Deletes all container images tagged with a PR suffix from GHCR.
# Required env vars: GH_TOKEN, OWNER, PACKAGE_NAME, PR_NUMBER
set -euo pipefail

TAG_SUFFIX="-pr-${PR_NUMBER}"
echo "Deleting images tagged with suffix: ${TAG_SUFFIX}"

VERSIONS=$(gh api \
  --paginate \
  --jq '.[]' \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/orgs/${OWNER}/packages/container/${PACKAGE_NAME}/versions")

PR_VERSIONS=$(echo "$VERSIONS" | jq -rs --arg suffix "$TAG_SUFFIX" '
  .[]
  | select(.metadata.container.tags | any(endswith($suffix)))
  | .id
')

if [ -z "$PR_VERSIONS" ]; then
  echo "No images found for PR #${PR_NUMBER}. Nothing to delete."
  exit 0
fi

COUNT=$(echo "$PR_VERSIONS" | grep -c .)
echo "Found ${COUNT} image(s) to delete."

DELETED=0
FAILED=0

for version_id in $PR_VERSIONS; do
  echo "Deleting version ID: ${version_id}"
  if gh api \
    --method DELETE \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/orgs/${OWNER}/packages/container/${PACKAGE_NAME}/versions/${version_id}"; then
    DELETED=$((DELETED + 1))
  else
    echo "WARNING: Failed to delete version ID ${version_id}"
    FAILED=$((FAILED + 1))
  fi
  sleep 1  # Avoid hitting the GitHub API rate limit
done

{
  echo "## PR Image Cleanup Summary"
  echo "- **PR:** #${PR_NUMBER}"
  echo "- **Deleted:** ${DELETED}"
  echo "- **Failed:** ${FAILED}"
} >> "$GITHUB_STEP_SUMMARY"

if [ "$FAILED" -gt 0 ]; then
  echo "Cleanup finished with ${FAILED} failure(s)."
  exit 1
fi

echo "PR image cleanup complete."
