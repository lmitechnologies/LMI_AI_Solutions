# Contributing

## Branch Model

The default branch is **`ais`**. All work targets `ais`.

## Workflow

1. **Create a feature branch** off the latest `ais`:

   ```bash
   git checkout ais && git pull
   git checkout -b feat/my-feature
   ```

2. **Develop & commit** on your branch — commit as often as you like locally.

3. **Before opening a PR, rebase onto `ais`** to keep history linear:

   ```bash
   git fetch origin
   git rebase origin/ais
   ```

   Resolve any conflicts, then force-push your branch:

   ```bash
   git push --force-with-lease
   ```

4. **Open a PR** targeting `ais`.

5. **PRs are squash-merged** into a single commit. The PR title becomes the commit message, so write it as a conventional commit (see below).

## Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) to drive automated semantic versioning and changelogs.

### Format

```
<type>(<scope>): <short summary>
```

### Types

| Type | When to use | Version bump |
|------|-------------|--------------|
| `feat` | New feature | minor |
| `fix` | Bug fix | patch |
| `docs` | Documentation only | — |
| `refactor` | Code change that neither fixes nor adds | — |
| `test` | Adding/updating tests | — |
| `chore` | Build, CI, tooling, deps | — |
| `perf` | Performance improvement | patch |

Add `!` after the type/scope for **breaking changes** → major bump:

```
feat!: remove deprecated v1 anomaly API
```

### Examples

```
feat(classifiers): add top-k accuracy metric
fix(od): handle empty predictions in NMS
docs: update installation guide
chore(deps): bump ultralytics to 8.3
```

## Code Quality

Pre-commit hooks run **Ruff** lint + format automatically on every commit. To run manually:

```bash
ruff check . --fix
ruff format .
```

Or run all hooks at once:

```bash
pre-commit run --all-files
```

## Quick Checklist

- [ ] Branch created from latest `ais`
- [ ] Rebased onto `origin/ais` before PR
- [ ] PR title follows conventional commit format
- [ ] Pre-commit hooks pass (`ruff check`, `ruff format`)
- [ ] Tests pass for affected modules
- [ ] No large files (> 500 KB) committed
