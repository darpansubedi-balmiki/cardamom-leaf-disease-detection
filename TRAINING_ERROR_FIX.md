# 🔧 Training Script Error Fix

## Your Error

```
(base) darpan@Darpans-Laptop-2 backend % python train.py
Traceback (most recent call last):
  File ".../backend/train.py", line 5, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named 'tqdm'
```

Or similar error for `matplotlib`.

---

## ✅ Quick Fix (3 Steps)

### Step 1: Pull Latest Changes

```bash
git pull
```

The requirements.txt has been updated to include training dependencies.

### Step 2: Reinstall Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This now includes:
- ✅ tqdm (progress bars)
- ✅ matplotlib (training plots)
- ✅ All other dependencies

### Step 3: Run Training

```bash
python train.py
```

**That's it!** ✅ Training should start with progress bars.

---

## What Was Fixed

The training script requires these packages that were missing from requirements.txt:

| Package | Purpose | Used In |
|---------|---------|---------|
| **tqdm** | Progress bars during training | Lines 13, 89, 131 |
| **matplotlib** | Training history plots | Lines 271-298 |

**Both are now included in requirements.txt** ✅

---

## Expected Output After Fix

When you run `python train.py`, you should see:

```
Using device: mps
Dataset path: dataset

Dataset loaded:
Training samples: 1206
Validation samples: 258
Classes: ['colletotrichum_blight', 'phyllosticta_leaf_spot', 'healthy']

Model created: EfficientNetV2-S

Starting training for 50 epochs...
============================================================

Epoch 1/50
------------------------------------------------------------
Training: 100%|████████████████| 38/38 [00:45<00:00,  1.20s/it, loss=1.0234, acc=45.67%]
Validation: 100%|██████████████| 9/9 [00:08<00:00,  1.05it/s]

Epoch 1 Results:
Train Loss: 1.0234 | Train Acc: 45.67%
Val Loss:   0.8912 | Val Acc:   52.33%

✓ Best model saved! (Val Acc: 52.33%)
...
```

With nice progress bars! 🎉

---

## If You Still Get Errors

### Error: "No module named 'torch'"

You need to install PyTorch first:

```bash
# For CPU or MPS (Apple Silicon)
pip install torch torchvision

# Or reinstall all dependencies
pip install -r requirements.txt
```

### Error: "CUDA out of memory"

Reduce batch size in train.py:

```python
BATCH_SIZE = 16  # Change from 32 to 16
```

### Error: Dataset not found

Make sure your dataset folder structure is:
```
backend/
  dataset/
    train/
      colletotrichum_blight/
      phyllosticta_leaf_spot/
      healthy/
    val/
      ...
```

---

## Summary

**Problem**: Missing tqdm and matplotlib in requirements.txt
**Solution**: Updated requirements.txt to include them
**Action**: `git pull && pip install -r requirements.txt`

**Your training should now work!** 🚀
