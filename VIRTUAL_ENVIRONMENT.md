# Virtual Environment Guide

## 🎯 What This User Discovered

A user figured out their issue:
> "i forgot to do this: source venv/bin/activate
> 
> Everything is working fine now"

**This was THE issue!** Without activating the virtual environment, packages were installing to the wrong location.

## What is a Virtual Environment?

A **virtual environment** (venv) is an isolated Python environment that keeps your project's dependencies separate from your system Python and other projects.

### Why Use Virtual Environments?

✅ **Isolation**: Each project has its own packages
✅ **No Conflicts**: Different projects can use different package versions
✅ **Reproducibility**: Easy to recreate exact environment
✅ **Clean**: Don't pollute system Python
✅ **No Permission Issues**: Install packages without sudo

### What Happens Without Venv?

❌ Packages install to system Python or wrong location
❌ Import errors even after pip install
❌ Conflicts with other projects
❌ Hard to reproduce environment
❌ May need admin/sudo permissions

## Creating a Virtual Environment

### macOS / Linux

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Or specify Python version
python3.11 -m venv venv
```

### Windows

```cmd
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
```

### Using Conda (Alternative)

```bash
conda create -n cardamom python=3.11
```

## Activating Virtual Environment

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows (Command Prompt)

```cmd
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
venv\Scripts\Activate.ps1
```

### Using Conda

```bash
conda activate cardamom
```

## How to Tell if Venv is Active

### 1. Shell Prompt Shows Venv Name

```bash
# Inactive:
user@computer:~/backend$

# Active:
(venv) user@computer:~/backend$
```

### 2. Check Python Location

```bash
# macOS/Linux
which python
# Should show: /path/to/backend/venv/bin/python

# Windows
where python
# Should show: C:\path\to\backend\venv\Scripts\python.exe
```

### 3. Test in Python

```python
import sys
print(sys.prefix)
# Should show venv path, not system Python
```

## Using Virtual Environment

### Complete Workflow

```bash
# 1. Create venv (one time only)
python -m venv venv

# 2. Activate venv (every time you work on project)
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install packages (one time, or when requirements change)
pip install -r requirements.txt

# 4. Work on your project
python train.py
python check_training_setup.py

# 5. Deactivate when done (optional)
deactivate
```

## Troubleshooting

### Issue 1: Forgot to Activate (This User's Issue!)

**Symptoms:**
- Installed packages but still get "No module named 'X'"
- pip install succeeds but imports fail
- check_training_setup.py shows packages missing

**Solution:**
```bash
# Activate venv FIRST
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Then try again
python check_training_setup.py
```

**How to Prevent:**
- Always activate venv before pip install
- Check shell prompt shows (venv)
- Add activation to your workflow

### Issue 2: Wrong Python Version

**Symptoms:**
- Created venv with Python 3.8 but need 3.11+
- Version check fails

**Solution:**
```bash
# Delete old venv
rm -rf venv

# Create with specific Python
python3.11 -m venv venv

# Activate and reinstall
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 3: Packages Not Found After Install

**Symptoms:**
- pip install succeeds
- But python can't find packages

**Possible Causes:**
1. Installed to wrong Python (not in venv)
2. Multiple Pythons on system
3. IDE using different Python

**Solution:**
```bash
# Verify venv is active
which python  # Should show venv path

# Reinstall in venv
source venv/bin/activate
pip install -r requirements.txt

# Verify installation
pip list | grep torch
```

### Issue 4: Multiple Virtual Environments

**Symptoms:**
- Confused which venv is active
- Packages in one venv but using another

**Solution:**
```bash
# Always deactivate before switching
deactivate

# Activate the correct one
source /path/to/correct/venv/bin/activate

# Verify
which python
pip list
```

### Issue 5: Permission Errors

**Symptoms:**
- "Permission denied" when creating venv
- Can't install packages

**Solution:**
```bash
# Don't use sudo with venv!
# Create venv in your home directory or project

# If in wrong location, create elsewhere
cd ~/projects/cardamom-detection/backend
python -m venv venv
```

### Issue 6: Corrupted Virtual Environment

**Symptoms:**
- Strange errors
- Packages behave oddly
- Imports fail randomly

**Solution:**
```bash
# Delete venv completely
rm -rf venv

# Create fresh venv
python -m venv venv

# Activate and reinstall
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue 7: IDE Not Using Venv

**Symptoms:**
- Terminal works but IDE doesn't
- VSCode/PyCharm shows import errors

**Solution:**

**VSCode:**
1. Open Command Palette (Cmd/Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose venv/bin/python

**PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing environment
3. Select venv/bin/python

## Best Practices

### 1. Always Use Virtual Environments

```bash
# For every Python project
cd my-project
python -m venv venv
source venv/bin/activate
```

### 2. One Venv Per Project

```
project1/
  └── venv/           ← Project 1's environment

project2/
  └── venv/           ← Project 2's environment
```

### 3. Add Venv to .gitignore

```gitignore
# Don't commit venv to git
venv/
env/
.venv/
```

### 4. Use requirements.txt

```bash
# Save installed packages
pip freeze > requirements.txt

# Recreate environment elsewhere
pip install -r requirements.txt
```

### 5. Activate Before Every Session

```bash
# Add to your workflow
cd backend
source venv/bin/activate  # Always do this first!
python train.py
```

## Quick Reference

### Common Commands

| Task | macOS/Linux | Windows |
|------|-------------|---------|
| Create venv | `python3 -m venv venv` | `python -m venv venv` |
| Activate | `source venv/bin/activate` | `venv\Scripts\activate` |
| Deactivate | `deactivate` | `deactivate` |
| Check if active | `which python` | `where python` |
| Install packages | `pip install -r requirements.txt` | Same |
| List packages | `pip list` | Same |
| Delete venv | `rm -rf venv` | `rmdir /s venv` |

### Venv Locations

```bash
# Python executable
venv/bin/python         # macOS/Linux
venv\Scripts\python.exe # Windows

# Installed packages
venv/lib/python3.X/site-packages/  # macOS/Linux
venv\Lib\site-packages\            # Windows

# Activation scripts
venv/bin/activate                  # macOS/Linux
venv\Scripts\activate.bat          # Windows CMD
venv\Scripts\Activate.ps1          # Windows PowerShell
```

## Summary

### The Key Lesson from This User

**Problem:** Installed packages but got import errors
**Cause:** Forgot to activate virtual environment
**Solution:** `source venv/bin/activate` before pip install

### Remember

```bash
# This is WRONG (no venv):
cd backend
pip install -r requirements.txt  ❌

# This is RIGHT (with venv):
cd backend
source venv/bin/activate         ✅
pip install -r requirements.txt  ✅
```

### Always Follow This Order

1. ✅ Create venv (once)
2. ✅ Activate venv (every session)
3. ✅ Install packages
4. ✅ Run scripts
5. ✅ Deactivate (when done)

## Need More Help?

- **Virtual environment issues?** Read this guide again
- **Installation problems?** See INSTALL_DEPENDENCIES.md
- **General questions?** See FAQ.md
- **Getting started?** See START_HERE.md

---

**🎉 Thanks to the user who discovered this issue and shared the solution!**

By documenting this, we help everyone avoid the same confusion.
