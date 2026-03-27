# Pre-commit Setup Guide

This repository uses [pre-commit](https://pre-commit.com/) with [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Hooks run automatically on every `git commit`.

**Ruff enforces:** code style, import sorting, common errors, and consistent formatting (line length 140, double quotes, rules E/F/I/B).

## Setup

```bash
pip install pre-commit
pre-commit install       # run once after cloning
```

## Typical Commit Workflow

```bash
git add <files>
git commit -m "your message"
```

If hooks fail, the commit is aborted. Ruff may auto-fix some issues (formatting, import order). Re-stage and retry:

```bash
git add -u          # stage the auto-fixed changes
git commit -m "your message"
```

For errors that require manual fixes, edit the flagged files per the error output, then:

```bash
git add <files>
git commit -m "your message"
```

## Other Useful Commands

```bash
pre-commit run --all-files          # check all files without committing
pre-commit run --files path/to/file.py  # check a specific file
pre-commit autoupdate               # update hooks to latest versions
```
