# Python AI Project — Setup Guide

## Prerequisites

- Python 3.8+ installed → [python.org](https://www.python.org/downloads/)
- `pip` available in your terminal

---

## 1. Create a Virtual Environment

A virtual environment keeps your project dependencies isolated from your system Python.

```bash
# Create the venv (only do this once)
python -m venv .venv
```

```bash
# Activate it — macOS / Linux
source .venv/bin/activate

# Activate it — Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate it — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

> You'll know it's active when you see `(.venv)` at the start of your terminal prompt.

---

## 2. Install Dependencies

With your venv active, install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 3. Run the Project

```bash
python main.py
```

> Replace `main.py` with whatever the entry point of the project is.

---

## 4. Adding New Packages

Install any new package as normal:

```bash
pip install numpy
```

Then **save it** to `requirements.txt` so others can replicate your environment:

```bash
pip freeze > requirements.txt
```

---

## 5. Deactivate the Virtual Environment

When you're done working:

```bash
deactivate
```

---

## Quick Reference

| Task                      | Command                           |
| ------------------------- | --------------------------------- |
| Create venv               | `python -m venv venv`             |
| Activate (Mac/Linux)      | `source venv/bin/activate`        |
| Activate (Windows)        | `venv\Scripts\activate.bat`       |
| Install from requirements | `pip install -r requirements.txt` |
| Save current packages     | `pip freeze > requirements.txt`   |
| Deactivate venv           | `deactivate`                      |

---

## Troubleshooting

**`python` not found** — try `python3` instead.

**PowerShell execution policy error** — run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Packages not found after activating** — make sure `(venv)` is visible in your prompt before running `pip install`.
