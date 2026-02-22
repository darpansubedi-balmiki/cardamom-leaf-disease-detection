# Frequently Asked Questions (FAQ)

## ❓ "I ran `git pull` but I'm still getting the same error!"

### The Answer

**`git pull` does NOT install Python packages!**

Here's what each command does:

```
┌────────────────────────────────────────┐
│  git pull                              │
│  └─→ Downloads: Code files, docs      │
│  └─→ Does NOT install packages!       │
│                                         │
│  pip install -r requirements.txt      │
│  └─→ Installs: PyTorch, NumPy, etc.   │
│  └─→ This fixes "No module" errors!   │
└────────────────────────────────────────┘
```

### What You Need to Do

After running `git pull`, you **MUST** also run:

```bash
cd backend
pip install -r requirements.txt
```

This installs the actual Python packages that the code needs.

### Why This Happens

- `git` manages **code files** (Python scripts, docs, configs)
- `pip` manages **Python packages** (libraries like PyTorch, NumPy)
- They are **separate systems** - one doesn't affect the other!

---

## ❓ "Do I need to install every time I pull?"

### Short Answer

**No**, only when `requirements.txt` changes.

### When to Reinstall

Install packages when:
- ✅ First time setting up
- ✅ `requirements.txt` file changed
- ✅ You see "No module named..." errors
- ✅ After switching branches (if requirements differ)

Skip installation when:
- ✅ Only documentation changed
- ✅ Only code logic changed
- ✅ requirements.txt unchanged

### How to Check if requirements.txt Changed

```bash
git pull
git log -1 --name-only
# Look for "backend/requirements.txt" in the output
```

Or check the commit message - we usually mention dependency changes.

---

## ❓ "How do I know if packages are installed correctly?"

### Quick Check

Run the diagnostic tool:

```bash
cd backend
python check_training_setup.py
```

Expected output when **correctly installed**:
```
✅ PyTorch              - OK
✅ torchvision          - OK
✅ NumPy                - OK
✅ tqdm                 - OK
✅ Matplotlib           - OK
✅ Pillow               - OK
```

### Manual Check

Or check imports directly:

```bash
python -c "import torch; import numpy; import matplotlib; print('All packages OK!')"
```

If you see "All packages OK!" - you're good!

---

## ❓ "I installed but still getting errors!"

### Common Issues & Solutions

**Issue 1: Wrong Directory**
```bash
# ❌ Wrong - in root directory
pip install -r requirements.txt

# ✅ Correct - in backend directory
cd backend
pip install -r requirements.txt
```

**Issue 2: Wrong Python Environment**
```bash
# Check which Python
which python
python --version

# If using virtual environment, activate it first:
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**Issue 3: Using Different Python**
```bash
# You installed with python3 but running with python
pip3 install -r requirements.txt  # Installed here
python train.py                    # But using this Python

# Solution: Use same Python for both
python3 -m pip install -r requirements.txt
python3 train.py
```

**Issue 4: Old pip**
```bash
# Upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ❓ "What's the difference between code files and packages?"

### Visual Explanation

| What | Managed By | Examples | Command |
|------|------------|----------|---------|
| **Code Files** | git | train.py, main.py, README.md | `git pull` |
| **Python Packages** | pip | torch, numpy, matplotlib | `pip install` |
| **Documentation** | git | *.md files, guides | `git pull` |
| **Dependencies** | pip | Listed in requirements.txt | `pip install` |

### Real Example

```bash
# Scenario: Missing PyTorch

# Step 1: Get latest code
git pull
# Result: You now have the latest train.py, README.md, etc.
# BUT: PyTorch is still not installed!

# Step 2: Install packages
cd backend
pip install -r requirements.txt
# Result: NOW PyTorch, NumPy, etc. are installed!

# Step 3: Verify
python check_training_setup.py
# Result: All ✅

# Step 4: Run training
python train.py
# Result: Works! 🎉
```

---

## ❓ "I forgot to activate my virtual environment!"

### The Discovery

A user figured out their issue:
> "i forgot to do this: source venv/bin/activate
> 
> Everything is working fine now"

This explains **ALL the confusion!** Without activating the virtual environment:
- Packages install to the wrong Python
- Imports fail even after pip install
- "Same issue" persists no matter what you try

### Symptoms

- ✅ `pip install` succeeds without errors
- ❌ `python script.py` shows "No module named 'torch'"
- ❌ `check_training_setup.py` shows packages missing
- ❓ Confused why packages "won't install"

### The Solution

```bash
# ALWAYS activate venv before installing or running scripts!

cd backend

# Create venv (if you haven't already)
python -m venv venv

# Activate venv - THIS IS THE KEY!
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Now install packages
pip install -r requirements.txt

# Now scripts will work
python train.py
```

### How to Tell if Venv is Active

Your shell prompt should show `(venv)`:

```bash
# NOT active:
user@computer:~/backend$

# Active (notice the (venv)):
(venv) user@computer:~/backend$
```

### Why Use Virtual Environments?

- ✅ Isolates project dependencies
- ✅ No conflicts with other projects
- ✅ No permission issues
- ✅ Easy to reproduce environment
- ✅ Clean and professional

### Learn More

See **VIRTUAL_ENVIRONMENT.md** for complete guide on virtual environments.

---

## ❓ "What if I'm still stuck?"

### Systematic Debugging Checklist

Try these in order:

1. **Check if virtual environment is active**
   ```bash
   # Should show (venv) in prompt
   # OR check Python location:
   which python  # macOS/Linux
   where python  # Windows
   # Should show venv path, not system Python
   ```

2. **Confirm you're in the right directory**
   ```bash
   pwd
   # Should show: .../cardamom-leaf-disease-detection/backend
   ```

3. **Check Python version**
   ```bash
   python --version
   # Should be Python 3.9-3.13
   ```

4. **Update pip**
   ```bash
   pip install --upgrade pip
   ```

5. **Clean reinstall**
   ```bash
   pip uninstall torch torchvision numpy matplotlib pillow opencv-python -y
   pip install -r requirements.txt
   ```

6. **Run diagnostic**
   ```bash
   python check_training_setup.py
   ```

6. **Check for errors in detail**
   ```bash
   python -c "import torch" 2>&1
   # Shows detailed error message
   ```

7. **Try automated setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### Still Not Working?

Check these resources:
- 📚 **INSTALL_DEPENDENCIES.md** - Complete installation guide
- 📚 **PYTHON_VERSION_GUIDE.md** - Python compatibility issues
- 📚 **TRAIN_PY_ERRORS.md** - Training-specific errors
- 📚 **PYTHON_313_FIX_SUMMARY.md** - Python 3.13 specific issues

---

## Quick Reference

| Problem | Solution |
|---------|----------|
| Missing packages after pull | `pip install -r requirements.txt` |
| "No module named..." | `pip install -r requirements.txt` |
| Installed but still errors | Check you're in `backend/` directory |
| Wrong Python version | Use Python 3.9-3.13 |
| Permission denied | Use `pip install --user ...` |
| Want automated setup | Run `./setup.sh` |
| Verify installation | Run `python check_training_setup.py` |
| Need fresh install | See INSTALL_DEPENDENCIES.md |

---

## Key Takeaway

```
┌─────────────────────────────────────────┐
│                                          │
│  git pull    → Gets code & docs          │
│  pip install → Installs packages         │
│                                          │
│  YOU NEED BOTH!                         │
│                                          │
└─────────────────────────────────────────┘
```

**After every `git pull`, check if you need to run `pip install`!**
