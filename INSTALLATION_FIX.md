# 🔧 Installation Fix - Python 3.13 on macOS ARM64

## Common Issues

### Issue 1: Pillow/pydantic build errors on Python 3.13
```
Getting requirements to build wheel did not run successfully
TypeError: ForwardRef._evaluate() missing required keyword-only argument
```

### Issue 2: Training script import errors
```
Traceback (most recent call last):
  File "train.py", line 13, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named 'tqdm'
```

## ✅ Solution (3 Steps)

### Step 1: Pull Latest Changes

The requirements.txt has been updated for Python 3.13 compatibility:

```bash
git pull
```

### Step 2: Update pip

```bash
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**That's it!** ✅ Installation should now work with all dependencies including training tools (tqdm, matplotlib).

---

## What Was Fixed

| Package | Old Version | New Version | Why |
|---------|-------------|-------------|-----|
| Pillow | 10.2.0 | >=10.4.0 | Python 3.13 wheels |
| pydantic | 2.5.3 | >=2.10.6 | Python 3.13 support |
| tqdm | ❌ Missing | >=4.66.0 | Training progress bars |
| matplotlib | ❌ Missing | >=3.8.0 | Training plots |

---

## If Step 3 Still Fails (Unlikely)

### Quick Fix: Install System Dependencies

```bash
brew install libjpeg zlib
pip install -r requirements.txt
```

### Alternative: Use Python 3.12 (Most Stable)

```bash
# Using pyenv
pyenv install 3.12.0
pyenv local 3.12.0
pip install -r requirements.txt

# OR using conda
conda create -n cardamom python=3.12
conda activate cardamom
cd backend
pip install -r requirements.txt
```

---

## What Was Fixed?

The requirements.txt was updated to use versions with Python 3.13 support:

| Package | Before | After |
|---------|--------|-------|
| Pillow | 10.2.0 | >=10.4.0 ✅ |
| pydantic | 2.5.3 | >=2.10.6 ✅ |
| torch | >=2.0.0 | >=2.1.0 ✅ |
| numpy | 1.26.3 | >=1.26.4 ✅ |

All packages now have pre-built wheels for Python 3.13 on macOS ARM64.

---

## Verify Installation

After installation succeeds, verify it works:

```bash
python -c "import torch, torchvision, PIL; print('✅ All packages imported successfully!')"
```

---

## Need More Help?

📚 **Complete guide**: [PYTHON_VERSION_GUIDE.md](PYTHON_VERSION_GUIDE.md)
- Supported Python versions (3.9-3.13)
- Platform-specific guides
- Detailed troubleshooting
- Virtual environment setup

🔄 **After installation**: [AFTER_PULL.md](AFTER_PULL.md)
- What to do next
- Verification steps
- Start using the system

---

## Quick Summary

✅ **Problem**: Pillow 10.2.0 and pydantic 2.5.3 don't have Python 3.13 wheels  
✅ **Solution**: Updated to Pillow >=10.4.0 and pydantic >=2.10.6 (Python 3.13 support)  
✅ **Action**: `git pull` then `pip install -r requirements.txt`  
✅ **Result**: Installation works on Python 3.13 macOS ARM64  

🎉 **You're all set!**
