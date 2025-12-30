# Pre-commit Setup Guide

This repository uses [pre-commit](https://pre-commit.com/) with [Ruff](https://github.com/astral-sh/ruff) as the primary linting and formatting backend to maintain code quality and consistency.

## What is Pre-commit?

Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. It runs automated checks on your code before each commit, ensuring that code quality standards are met before changes enter the repository.

## What is Ruff?

Ruff is an extremely fast Python linter and formatter written in Rust. It replaces multiple tools like Flake8, isort, and Black with a single, high-performance tool that can check and format your code in milliseconds.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Pre-commit

```bash
pip install pre-commit
```

### Install the Git Hook Scripts

After cloning this repository, navigate to the project root and run:

```bash
pre-commit install
```

This installs the pre-commit hook into your `.git/hooks/` directory. The hooks will now run automatically on `git commit`.

## Usage

### Automatic Usage

Once installed, pre-commit runs automatically every time you commit:

```bash
git add .
git commit -m "Your commit message"
```

If any checks fail, the commit will be aborted and you'll see which files need fixing.

### Manual Usage

You can run pre-commit manually on all files:

```bash
pre-commit run --all-files
```

Or on specific files:

```bash
pre-commit run --files path/to/file.py
```

## Common Workflows

### First-Time Setup

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Run on all files to check current state
pre-commit run --all-files
```

### Skipping Hooks (Use Sparingly)

If you need to commit without running hooks (not recommended):

```bash
git commit --no-verify -m "Emergency fix"
```

### Updating Hooks

To update all hooks to their latest versions:

```bash
pre-commit autoupdate
```

## What Ruff Checks

Ruff performs comprehensive linting that includes:

- Code style violations (similar to Flake8, pycodestyle)
- Import sorting and organization (similar to isort)
- Common programming errors and anti-patterns
- Code complexity checks
- Docstring conventions
- Security issues
- And many more rules from popular Python linters

Ruff also formats your code (similar to Black) to ensure consistent style across the codebase.

## Handling Failures

When pre-commit fails:

1. **Review the output**: Pre-commit will show which files failed and why
2. **Auto-fixed issues**: Some issues (like formatting) are fixed automatically. Stage the changes and commit again:
   ```bash
   git add .
   git commit -m "Your message"
   ```
3. **Manual fixes required**: For issues that can't be auto-fixed, modify the code according to the error messages
4. **Re-commit**: After fixing issues, stage and commit again

## Configuration

The pre-commit configuration is stored in `.pre-commit-config.yaml` at the repository root. Ruff-specific settings can be found in `pyproject.toml`.

## VS Code Integration

### Install Ruff Extension

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X on Mac)
3. Search for "Ruff"
4. Install the official extension by Astral Software (charliermarsh.ruff)

### Configure VS Code Settings

Add the following to your VS Code settings (`.vscode/settings.json` in the repository or your user settings):

```json
{
  // Enable Ruff as the default formatter
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  // Enable Ruff linting
  "ruff.enable": true,
  "ruff.lint.enable": true,
  "ruff.format.enable": true
}
```

### Benefits of VS Code Integration

- **Real-time feedback**: See linting errors as you type
- **Auto-fix on save**: Automatically fix issues when saving files
- **Import organization**: Automatically sort and organize imports
- **Consistent formatting**: Match the pre-commit hook behavior in your editor

This ensures your code is checked and formatted in VS Code the same way pre-commit will check it before commits.

## Troubleshooting

### Pre-commit doesn't run
- Ensure you've run `pre-commit install`
- Check that you're in the repository root directory

### Hooks are slow
- Ruff is extremely fast, but first runs may take longer
- Consider updating to the latest version: `pre-commit autoupdate`

### Conflicts with IDE formatting
- Configure your IDE to use Ruff for formatting
- Ensure IDE settings match the project's Ruff configuration

### Need to bypass temporarily
```bash
git commit --no-verify
```
Use this sparingly and only when absolutely necessary.

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ruff Rules Reference](https://docs.astral.sh/ruff/rules/)