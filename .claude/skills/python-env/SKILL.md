---
name: python-env
description: Use this skill when running Python scripts, installing packages, or executing any Python-related commands. Always use the project virtual environment and PowerShell syntax.
---

# Python Environment Skill

## Rules

- **Always** use the virtual environment Python at `C:\Users\yijun.jiang\projects\myenv\Scripts\python.exe`
- **Always** use Windows PowerShell syntax for commands (not bash/Unix syntax)
- Use `C:\Users\yijun.jiang\projects\myenv\Scripts\pip.exe` for package installation

## Command Examples

**Run a Python script:**
```powershell
C:\Users\yijun.jiang\projects\myenv\Scripts\python.exe script.py
```

**Run a Python module:**
```powershell
C:\Users\yijun.jiang\projects\myenv\Scripts\python.exe -m pytest
C:\Users\yijun.jiang\projects\myenv\Scripts\python.exe -m pip install <package>
```

**Install a package:**
```powershell
C:\Users\yijun.jiang\projects\myenv\Scripts\pip.exe install <package>
```

**Check installed packages:**
```powershell
C:\Users\yijun.jiang\projects\myenv\Scripts\pip.exe list
```

**Run inline Python:**
```powershell
C:\Users\yijun.jiang\projects\myenv\Scripts\python.exe -c "import sys; print(sys.version)"
```

## Key Points

- Never use `python`, `python3`, or `pip` bare commands — always use the full path
- Never activate the venv with `source` (that's Unix syntax); use the full path instead
- PowerShell uses `;` not `&&` to chain commands, and `$env:VAR` for environment variables
