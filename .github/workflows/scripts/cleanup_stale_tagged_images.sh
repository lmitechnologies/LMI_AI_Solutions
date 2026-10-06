#!/usr/bin/env bash
# Deletes tagged container images from GHCR unless a tag is in ALLOWED_TAGS or is the PR tag of an open PR.
# Required env vars: GH_TOKEN, GH_REPO, OWNER, PACKAGE_NAME, ALLOWED_TAGS (whitespace-separated)
# Optional env vars: DRY_RUN (default: false)
set -euo pipefail

DRY_RUN=${DRY_RUN:-false}

ALLOWED=$(jq -nc --arg tags "$ALLOWED_TAGS" '[$tags | scan("\\S+")]')
if [ "$(echo "$ALLOWED" | jq length)" -eq 0 ]; then
  echo "ERROR: ALLOWED_TAGS is empty"
  exit 1
fi
OPEN_PRS=$(gh pr list --state open --limit 1000 --json number --jq '[.[].number]')
echo "Allowed tags: ${ALLOWED}"
echo "Open PRs: ${OPEN_PRS}"
echo "Dry run: ${DRY_RUN}"

VERSIONS=$(gh api \
  --paginate \
  --jq '.[]' \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/orgs/${OWNER}/packages/container/${PACKAGE_NAME}/versions")

STALE=$(echo "$VERSIONS" | jq -rs --argjson allowed "$ALLOWED" --argjson open "$OPEN_PRS" '
  def keep: IN($allowed[]) or ([capture("-pr-(?<n>[0-9]+)$").n | tonumber] | any(IN($open[])));
  .[]
  | select(.metadata.container.tags | length > 0)
  | select(.metadata.container.tags | any(keep) | not)
  | "\(.id)\t\(.metadata.container.tags | join(","))"
')

if [ -z "$STALE" ]; then
  echo "No stale tagged images found. Nothing to delete."
  exit 0
fi

COUNT=$(echo "$STALE" | grep -c .)
echo "Found ${COUNT} stale tagged image(s)."

DELETED=0
FAILED=0

while IFS=$'\t' read -r version_id tags; do
  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would delete version ID: ${version_id} (${tags})"
  else
    echo "Deleting version ID: ${version_id} (${tags})"
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
done <<< "$STALE"

{
  echo "## Stale Tagged Image Cleanup Summary"
  echo "- **Package:** \`${PACKAGE_NAME}\`"
  echo "- **Dry run:** ${DRY_RUN}"
  echo "- **Candidates found:** ${COUNT}"
  if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "### Candidates (dry run)"
    while IFS=$'\t' read -r version_id tags; do
      echo "- \`${version_id}\`: ${tags}"
    done <<< "$STALE"
  else
    echo "- **Deleted:** ${DELETED}"
    echo "- **Failed:** ${FAILED}"
  fi
} >> "$GITHUB_STEP_SUMMARY"

if [ "$DRY_RUN" != "true" ] && [ "$FAILED" -gt 0 ]; then
  echo "Cleanup finished with ${FAILED} failure(s)."
  exit 1
fi

echo "Stale tagged image cleanup complete."
