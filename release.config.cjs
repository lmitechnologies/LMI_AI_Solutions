// RELEASE_MAJOR=true (set by the Release workflow input) forces a major bump; otherwise breaking changes bump minor.
const releaseRules = process.env.RELEASE_MAJOR === "true" ? [{ release: "major" }] : [{ breaking: true, release: "minor" }];

module.exports = {
  branches: ["ais"],
  plugins: [
    ["@semantic-release/commit-analyzer", { preset: "conventionalcommits", releaseRules }],
    ["@semantic-release/release-notes-generator", { preset: "conventionalcommits" }],
    "@semantic-release/changelog",
    [
      "@semantic-release/exec",
      {
        prepareCmd:
          "uv version ${nextRelease.version} --no-sync && bash .github/workflows/scripts/rotate_prerelease.sh ${nextRelease.gitTag}",
        successCmd: "bash .github/workflows/scripts/publish_release_notes.sh ${nextRelease.gitTag}",
      },
    ],
    "@semantic-release/github",
    [
      "@semantic-release/git",
      {
        assets: ["CHANGELOG.md", "pyproject.toml", "uv.lock", "PRERELEASE.md", "docs/releases/*.md"],
        message: "chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}",
      },
    ],
  ],
};
