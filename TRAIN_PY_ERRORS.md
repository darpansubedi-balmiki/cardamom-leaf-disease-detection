# train.py Error Troubleshooting Guide

## Quick Diagnosis Tool

Before training, **always run the environment checker first**:

```bash
cd backend
python check_training_setup.py
```

This will check:
- ✅ All required packages installed
- ✅ PyTorch and device compatibility
- ✅ Dataset structure
- ✅ Output directories

## Common Errors and Solutions

### Error: Import Errors (Line 5-13)

**Symptoms:**
```
Traceback (most recent call last):
  File "train.py", line 5, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

**Or:**
```
ModuleNotFoundError: No module named 'tqdm'
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution:**
```bash
# Pull latest changes (includes updated requirements.txt)
git pull

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python check_training_setup.py
```

**Why this happens:**
- Missing packages in virtual environment
- requirements.txt not updated (missing tqdm/matplotlib)
- Wrong Python environment activated

---

### Error: Dataset Directory Not Found

**Symptoms:**
```
❌ ERROR: Dataset directory not found!
Expected: /path/to/backend/dataset
```

**Solution:**

**Option 1: You have images but haven't organized them**
```bash
# Put your images in backend/ directory like:
# colletotrichum_blight/*.jpg
# phyllosticta_leaf_spot/*.jpg
# healthy/*.jpg

# Then run the split script
python split_dataset.py
```

**Option 2: Dataset is in wrong location**
```bash
# Make sure you're in the backend directory
cd backend
python train.py
```

**Option 3: No data yet**
- See TRAINING_YOUR_MODEL.md for dataset collection guidance
- Need minimum 500 images per class (1,500 total)

---

### Error: MPS/Device Issues (macOS)

**Symptoms:**
```
RuntimeError: MPS backend out of memory
RuntimeError: MPS does not support...
```

**Solution 1: Force CPU**

Edit `train.py` line 67-68:
```python
# Change this:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# To this (force CPU):
DEVICE = torch.device("cpu")
```

**Solution 2: Reduce batch size**

Edit `train.py` line 60:
```python
BATCH_SIZE = 16  # Changed from 32
```

**Solution 3: Update PyTorch**
```bash
pip install --upgrade torch torchvision
```

---

### Error: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

**Reduce batch size** - Edit `train.py`:
```python
BATCH_SIZE = 16  # Or even 8 if still failing
```

**Or reduce image size:**
```python
IMG_SIZE = 128  # Changed from 224
```

---

### Error: Empty Dataset

**Symptoms:**
```
RuntimeError: Found 0 files in subfolders of: dataset/train/colletotrichum_blight
```

**Solution:**

Your dataset folders exist but are empty!

```bash
# Check what's in your directories
ls -la dataset/train/colletotrichum_blight/
ls -la dataset/train/phyllosticta_leaf_spot/
ls -la dataset/train/healthy/

# If empty, you need to run split_dataset.py
# First, make sure source images are in backend/:
ls -la colletotrichum_blight/
ls -la phyllosticta_leaf_spot/
ls -la healthy/

# Then split them
python split_dataset.py
```

---

### Error: Permission Denied (models directory)

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'models/cardamom_model.pt'
```

**Solution:**
```bash
# Make sure models directory has write permissions
chmod 755 models/
# Or remove and recreate
rm -rf models
mkdir models
```

---

### Error: Incomplete Error Message

**If you see:**
```
Traceback (most recent call last):
  File "train.py", line 5, in <module>
```

**But no actual error message**, this usually means:

1. **Terminal output was cut off** - Run again with more verbose output:
   ```bash
   python train.py 2>&1 | tee training.log
   ```

2. **Python version issue** - Check Python version:
   ```bash
   python --version
   # Should be 3.9-3.13
   ```

3. **Environment issue** - Run diagnostics:
   ```bash
   python check_training_setup.py
   ```

---

## Step-by-Step Troubleshooting

If you're getting errors, follow these steps in order:

### Step 1: Check Environment
```bash
cd backend
python check_training_setup.py
```

Fix any issues it reports before proceeding.

### Step 2: Verify Requirements
```bash
# Make sure you have latest code
git pull

# Install/update requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Check Dataset
```bash
# Verify structure
ls dataset/train/
ls dataset/val/
ls dataset/test/

# Each should have 3 subdirectories:
# - colletotrichum_blight
# - phyllosticta_leaf_spot
# - healthy

# Check image counts
find dataset/train -name "*.jpg" -o -name "*.png" | wc -l
# Should be > 500
```

### Step 4: Test Import
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')"
```

### Step 5: Try Training
```bash
python train.py
```

---

## Getting Help

If you're still stuck after trying the above:

1. **Capture full error output:**
   ```bash
   python train.py 2>&1 | tee error.log
   ```

2. **Check environment info:**
   ```bash
   python check_training_setup.py > setup_info.txt
   ```

3. **Include in your question:**
   - Python version (`python --version`)
   - Operating system (macOS/Linux/Windows)
   - GPU/CPU info
   - Full error message from error.log
   - Output from check_training_setup.py

---

## Quick Reference

| Error Type | Quick Fix |
|------------|-----------|
| Import error | `pip install -r requirements.txt` |
| Dataset not found | `python split_dataset.py` |
| MPS error | Edit train.py, force CPU |
| CUDA OOM | Reduce BATCH_SIZE to 16 |
| Empty dataset | Check source images, re-run split |
| Permission error | `chmod 755 models/` |

---

## Prevention

**Before every training session:**

1. ✅ Run: `python check_training_setup.py`
2. ✅ Verify dataset has images
3. ✅ Check available disk space (need ~5GB)
4. ✅ Check available memory (need ~4GB RAM minimum)

**This will save you debugging time!** 🎯
