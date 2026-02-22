# Installing Dependencies

## Quick Answer

If you see this error:
```
❌ Error importing PyTorch: No module named 'torch'
```

**Run this command:**
```bash
cd backend
pip install -r requirements.txt
```

That's it! Continue reading for details and troubleshooting.

---

## Prerequisites

- **Python 3.9 - 3.13** (Check: `python --version`)
- **pip** (Python package installer)
- **5-10 minutes** for installation
- **~160MB** disk space for dependencies

---

## Installation Methods

### Method 1: Manual (Recommended)

**Step-by-step installation:**

```bash
# 1. Navigate to backend directory
cd backend

# 2. (Optional but recommended) Upgrade pip
pip install --upgrade pip

# 3. Install all dependencies
pip install -r requirements.txt
```

**Expected output:**
```
Collecting fastapi==0.109.1
Collecting torch>=2.1.0
Collecting pillow>=10.4.0
...
Successfully installed [11 packages]
```

**Installation time:** 3-5 minutes (depending on internet speed)

### Method 2: Automated (Quick)

Use the setup script:

```bash
cd backend
chmod +x setup.sh
./setup.sh
```

The script will:
1. Check Python version
2. Upgrade pip
3. Install dependencies
4. Run diagnostic check
5. Report success/failure

---

## What Gets Installed

**11 packages (~160MB total):**

### API Framework
- **fastapi** (0.109.1) - Fast web framework for ML API
- **uvicorn** (0.27.0) - ASGI server
- **python-multipart** (0.0.22) - File upload handling
- **pydantic** (≥2.10.6) - Data validation

### Machine Learning
- **torch** (≥2.1.0, ~79MB) - PyTorch deep learning framework
- **torchvision** (≥0.16.0, ~1.9MB) - Computer vision models and transforms

### Image Processing
- **Pillow** (≥10.4.0, ~4.7MB) - Image file handling
- **opencv-python** (≥4.10.0.84, ~46MB) - Computer vision operations
- **numpy** (≥1.26.4) - Numerical computing

### Training Tools
- **tqdm** (≥4.66.0, ~108KB) - Progress bars during training
- **matplotlib** (≥3.8.0, ~391KB) - Training history plots

---

## Verification

After installation, verify everything works:

```bash
# Run diagnostic check
python check_training_setup.py
```

**Expected output:**
```
============================================================
TRAINING ENVIRONMENT CHECK
============================================================

✅ PyTorch              - OK
✅ torchvision          - OK
✅ NumPy                - OK
✅ tqdm                 - OK
✅ Matplotlib           - OK
...
✅ ALL CHECKS PASSED!
```

---

## Troubleshooting

### Issue 1: Python Version Too Old

**Error:**
```
ERROR: Package requires Python >=3.9
```

**Solution:**
Upgrade Python to 3.9 or higher:
- **macOS:** `brew upgrade python` or download from python.org
- **Linux:** `sudo apt update && sudo apt install python3.9`
- **Windows:** Download installer from python.org

### Issue 2: pip Too Old

**Error:**
```
WARNING: pip is out of date
```

**Solution:**
```bash
pip install --upgrade pip
```

### Issue 3: Permission Denied

**Error:**
```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Install for current user only
pip install --user -r requirements.txt
```

### Issue 4: Network Timeout

**Error:**
```
ReadTimeoutError: HTTPSConnectionPool... Read timed out
```

**Solution:**
```bash
# Increase timeout
pip install --timeout=120 -r requirements.txt

# Or try again later
```

### Issue 5: Platform-Specific Issues

**macOS ARM (M1/M2):**
- torch and torchvision have native Apple Silicon support
- Uses MPS (Metal Performance Shaders) for GPU acceleration

**Linux:**
- May need system libraries: `sudo apt install python3-dev libopencv-dev`

**Windows:**
- May need Visual C++ redistributables
- Download from Microsoft's website

### Issue 6: Virtual Environment Issues

If using a virtual environment, make sure it's activated:

```bash
# Create venv
python -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Then install
pip install -r requirements.txt
```

### Issue 7: Clean Reinstall

If installations are corrupted:

```bash
# Uninstall all
pip uninstall -y -r requirements.txt

# Clear cache
pip cache purge

# Reinstall
pip install -r requirements.txt
```

---

## After Installation

**Next steps:**

1. **Verify installation:**
   ```bash
   python check_training_setup.py
   ```

2. **Check if you have training data:**
   - Should see: `dataset/train/`, `dataset/val/`, `dataset/test/`
   - If not, see YOURE_READY.md or NEXT_STEPS.md

3. **Start training (if you have data):**
   ```bash
   python train.py
   ```

4. **Or start the API:**
   ```bash
   uvicorn app.main:app --reload
   ```

---

## Platform-Specific Notes

### macOS
- **Installation time:** 3-5 minutes
- **Special:** MPS acceleration available on M1/M2
- **Brew packages:** Usually not needed

### Linux
- **Installation time:** 3-5 minutes
- **May need:** `sudo apt install python3-dev python3-pip`
- **CUDA:** If you have NVIDIA GPU, torch will auto-detect

### Windows
- **Installation time:** 5-10 minutes
- **May need:** Microsoft Visual C++ 14.0+ redistributables
- **PowerShell:** Use PowerShell or Command Prompt, not Git Bash

---

## Success Checklist

After running `pip install -r requirements.txt`:

- [ ] Installation completed without errors
- [ ] All 11 packages installed
- [ ] `python check_training_setup.py` passes all checks
- [ ] No import errors when running `python train.py` or API
- [ ] Training or API can start successfully

---

## Need More Help?

- **Installation issues:** See INSTALLATION_FIX.md
- **Training errors:** See TRAIN_PY_ERRORS.md
- **Python version issues:** See PYTHON_VERSION_GUIDE.md
- **After installation:** See AFTER_PULL.md
- **All documentation:** See DOCUMENTATION_INDEX.md

---

## Summary

**Quick command:**
```bash
cd backend && pip install -r requirements.txt
```

**Time:** 3-5 minutes  
**Size:** ~160MB  
**Packages:** 11  
**Result:** Ready to train! 🚀
