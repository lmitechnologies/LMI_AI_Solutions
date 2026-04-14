#!/usr/bin/env bash
# Deletes untagged container images older than RETENTION_DAYS from GHCR.
# Required env vars: GH_TOKEN, OWNER, PACKAGE_NAME
# Optional env vars: RETENTION_DAYS (default: 7), DRY_RUN (default: false)
set -euo pipefail

RETENTION_DAYS=${RETENTION_DAYS:-7}
DRY_RUN=${DRY_RUN:-false}

# GitHub Actions 'number' inputs may arrive as floats (e.g. "7.0");
# truncate to integer for shell arithmetic and date command.
RETENTION_DAYS=${RETENTION_DAYS%.*}

# Validate retention period
if [ "${RETENTION_DAYS}" -lt 1 ]; then
  echo "ERROR: RETENTION_DAYS must be >= 1, got: ${RETENTION_DAYS}"
  exit 1
fi

# Compute cutoff as epoch seconds for reliable numeric comparison
CUTOFF_EPOCH=$(date -d "${RETENTION_DAYS} days ago" +%s)
CUTOFF_ISO=$(date -d "@${CUTOFF_EPOCH}" -Iseconds)
echo "Retention: ${RETENTION_DAYS} days — deleting untagged images older than ${CUTOFF_ISO}"
echo "Dry run: ${DRY_RUN}"

# Fetch all pages; --paginate emits concatenated JSON arrays so we
# stream individual objects with --jq to get valid input for slurp
VERSIONS=$(gh api \
  --paginate \
  --jq '.[]' \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/orgs/${OWNER}/packages/container/${PACKAGE_NAME}/versions")

# Filter: untagged + older than cutoff (compare epoch timestamps)
OLD_VERSIONS=$(echo "$VERSIONS" | jq -rs --argjson cutoff "$CUTOFF_EPOCH" '
  .[]
  | select(.metadata.container.tags | length == 0)
  | select((.created_at | fromdateiso8601) < $cutoff)
  | .id
')

if [ -z "$OLD_VERSIONS" ]; then
  echo "No untagged images older than ${RETENTION_DAYS} days found. Nothing to delete."
  exit 0
fi

COUNT=$(echo "$OLD_VERSIONS" | grep -c .)
echo "Found ${COUNT} untagged image(s) to delete."

DELETED=0
FAILED=0

for version_id in $OLD_VERSIONS; do
  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would delete version ID: ${version_id}"
  else
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
  fi
done

{
  echo "## Cleanup Summary"
  echo "- **Package:** \`${PACKAGE_NAME}\`"
  echo "- **Retention:** ${RETENTION_DAYS} days"
  echo "- **Dry run:** ${DRY_RUN}"
  echo "- **Candidates found:** ${COUNT}"
  if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "### Candidates (dry run)"
    for version_id in $OLD_VERSIONS; do
      echo "- \`${version_id}\`"
    done
  else
    echo "- **Deleted:** ${DELETED}"
    echo "- **Failed:** ${FAILED}"
  fi
} >> "$GITHUB_STEP_SUMMARY"

if [ "$DRY_RUN" != "true" ] && [ "$FAILED" -gt 0 ]; then
  echo "Cleanup finished with ${FAILED} failure(s)."
  exit 1
fi

echo "Cleanup complete."
