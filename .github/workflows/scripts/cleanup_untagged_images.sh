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

# Tags point to image indexes whose child manifests are listed as untagged versions; deleting those breaks the tag.
REGISTRY_TOKEN=$(curl -fsS -u "${OWNER}:${GH_TOKEN}" \
  "https://ghcr.io/token?scope=repository:${OWNER}/${PACKAGE_NAME}:pull" | jq -r .token)
ACCEPT="application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json"
ACCEPT+=",application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json"
TAGGED_DIGESTS=$(echo "$VERSIONS" | jq -rs '.[] | select(.metadata.container.tags | length > 0) | .name')
CHILD_DIGESTS=""
for digest in $TAGGED_DIGESTS; do
  MANIFEST=$(curl -fsS -H "Authorization: Bearer ${REGISTRY_TOKEN}" -H "Accept: ${ACCEPT}" \
    "https://ghcr.io/v2/${OWNER}/${PACKAGE_NAME}/manifests/${digest}")
  CHILD_DIGESTS+=$(echo "$MANIFEST" | jq -r '.manifests[]?.digest')$'\n'
done
PROTECTED=$(printf '%s' "$CHILD_DIGESTS" | jq -Rsc 'split("\n") | map(select(length > 0))')
echo "Protected child manifests of tagged images: $(echo "$PROTECTED" | jq length)"

# Filter: untagged + not a child of a tagged image + older than cutoff (compare epoch timestamps)
OLD_VERSIONS=$(echo "$VERSIONS" | jq -rs --argjson cutoff "$CUTOFF_EPOCH" --argjson protected "$PROTECTED" '
  .[]
  | select(.metadata.container.tags | length == 0)
  | select(.name | IN($protected[]) | not)
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
