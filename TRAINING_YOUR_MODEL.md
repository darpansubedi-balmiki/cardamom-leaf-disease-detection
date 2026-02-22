# Training Your Cardamom Disease Detection Model 🎓

## Great News! Your Data is Ready! 🎉

You have successfully collected and prepared your dataset:

### 📊 Dataset Statistics

**Total Images: 1,724**

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Colletotrichum Blight | 196 | 42 | 42 | 280 |
| Phyllosticta Leaf Spot | 464 | 99 | 100 | 663 |
| Healthy | 546 | 117 | 118 | 781 |
| **TOTAL** | **1,206** | **258** | **260** | **1,724** |

**Split Ratio**: 70% train / 15% val / 15% test ✓

### ✅ What's Ready
- ✅ Dataset collected and organized
- ✅ Train/val/test splits created
- ✅ Training script (`train.py`) configured
- ✅ Model architecture (EfficientNetV2-S) ready
- ✅ Data augmentation configured

---

## 🚀 Step-by-Step Training Guide

### Step 1: Install Dependencies

First, make sure you have all required packages:

```bash
cd backend
pip install -r requirements.txt
```

> ✅ All training dependencies now included in requirements.txt (tqdm, matplotlib, etc.)

### Step 2: Verify Your Dataset

Check that your dataset is properly organized:

```bash
cd backend
python split_dataset.py --check  # Optional verification
```

Your dataset structure should look like:
```
backend/
├── dataset/
│   ├── train/
│   │   ├── colletotrichum_blight/  (196 images)
│   │   ├── phyllosticta_leaf_spot/ (464 images)
│   │   └── healthy/                (546 images)
│   ├── val/
│   │   ├── colletotrichum_blight/  (42 images)
│   │   ├── phyllosticta_leaf_spot/ (99 images)
│   │   └── healthy/                (117 images)
│   └── test/
│       ├── colletotrichum_blight/  (42 images)
│       ├── phyllosticta_leaf_spot/ (100 images)
│       └── healthy/                (118 images)
```

### Step 3: Start Training! 🎯

Now you're ready to train:

```bash
cd backend
python train.py
```

**What will happen:**
1. EfficientNetV2-S model will be initialized with ImageNet weights
2. Training will start for up to 50 epochs
3. Progress bars will show training/validation progress
4. Best model will be saved automatically
5. Training history plot will be generated

**Expected Training Time:**
- **With GPU**: 30-60 minutes
- **With CPU**: 3-6 hours
- **With Apple Silicon (MPS)**: 1-2 hours

### Step 4: Monitor Training

Watch for these key metrics:

**Good Signs:**
- ✅ Training loss decreasing steadily
- ✅ Validation accuracy increasing
- ✅ Train and val accuracy close together (gap < 10%)
- ✅ Reaching 85%+ validation accuracy

**Warning Signs:**
- ⚠️ Validation accuracy much lower than training (overfitting)
- ⚠️ Both accuracies stuck at 33% (not learning)
- ⚠️ Loss increasing (learning rate too high)

**Training Output Example:**
```
Using device: cuda
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
Training: 100%|████████| 38/38 [00:45<00:00, loss=0.8234, acc=65.42%]
Validation: 100%|████████| 9/9 [00:05<00:00]

Epoch 1 Results:
Train Loss: 0.8234 | Train Acc: 65.42%
Val Loss:   0.6821 | Val Acc:   72.48%
✓ Best model saved! (Val Acc: 72.48%)

[... continues for more epochs ...]

Epoch 25/50
------------------------------------------------------------
Training: 100%|████████| 38/38 [00:42<00:00, loss=0.1234, acc=95.86%]
Validation: 100%|████████| 9/9 [00:04<00:00]

Epoch 25 Results:
Train Loss: 0.1234 | Train Acc: 95.86%
Val Loss:   0.2156 | Val Acc:   92.64%
✓ Best model saved! (Val Acc: 92.64%)
```

### Step 5: Check Training Results

After training completes, you'll have:

1. **Trained Model**: `backend/models/cardamom_model.pt`
2. **Training Plot**: `backend/training_history.png`

Open `training_history.png` to visualize:
- Loss curves (should decrease)
- Accuracy curves (should increase)

### Step 6: Evaluate Model Performance

Create a test evaluation:

```bash
cd backend
python evaluate.py
```

This will show:
- Accuracy per class
- Confusion matrix
- Precision, recall, F1-score

**Expected Results:**
```
Classification Report:
                           precision    recall  f1-score   support

     Colletotrichum Blight     0.88      0.90      0.89        42
  Phyllosticta Leaf Spot        0.92      0.91      0.91       100
                Healthy          0.95      0.94      0.94       118

              accuracy                           0.92       260
             macro avg          0.92      0.92      0.92       260
          weighted avg          0.92      0.92      0.92       260
```

### Step 7: Deploy Your Trained Model

Once you're satisfied with the results:

1. **The model is already in the right place!**
   - Location: `backend/models/cardamom_model.pt`
   - This is where the API looks for it

2. **Restart your backend server:**
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Verify model is loaded:**
   Check the console output for:
   ```
   ✓ Loaded trained model weights from models/cardamom_model.pt
   Model is trained: True
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```
   
   Should return:
   ```json
   {
     "status": "ok",
     "model_status": "trained"
   }
   ```

5. **Test with real image:**
   Upload an image through the web app or mobile app!

---

## 📈 Expected Performance

With your dataset (1,724 images), you should achieve:

### Minimum Acceptable:
- **Validation Accuracy**: 85%+
- **Test Accuracy**: 80%+
- **Per-class Accuracy**: >75% for each class

### Good Performance:
- **Validation Accuracy**: 90%+
- **Test Accuracy**: 85%+
- **Per-class Accuracy**: >85% for each class

### Excellent Performance:
- **Validation Accuracy**: 95%+
- **Test Accuracy**: 90%+
- **Per-class Accuracy**: >90% for each class

---

## 🛠️ Troubleshooting

### Problem: "No module named 'torchvision'"
**Solution:**
```bash
pip install torch torchvision
```

### Problem: Training very slow on CPU
**Solution:**
- Consider using Google Colab (free GPU)
- Or reduce batch size in train.py: `BATCH_SIZE = 16`

### Problem: "CUDA out of memory"
**Solution:**
- Reduce batch size: `BATCH_SIZE = 16` or `8`
- Close other GPU-using programs

### Problem: Validation accuracy stuck at ~33%
**Solution:**
- Model is not learning (random guessing for 3 classes)
- Check that:
  - Images are loaded correctly
  - Labels are correct
  - Learning rate is not too high or too low
  - Dataset is balanced enough

### Problem: Overfitting (train 95%, val 70%)
**Solution:**
- More data augmentation
- Increase dropout
- Early stopping is working (it will save best val accuracy)
- Consider collecting more data

### Problem: Class imbalance warning
**Current imbalance:**
- Colletotrichum Blight: 280 images (16%)
- Phyllosticta: 663 images (38%)  
- Healthy: 781 images (45%)

**If needed:**
```python
# Add weighted loss in train.py
from torch.nn import CrossEntropyLoss
weights = torch.tensor([2.79, 1.0, 1.0])  # Weight for rare class
criterion = CrossEntropyLoss(weight=weights)
```

---

## 🎯 After Training Success

Once your model is trained and deployed:

### Immediate Actions:
1. ✅ Test with real images via web app
2. ✅ Test with mobile app
3. ✅ Verify predictions are accurate
4. ✅ Check confidence scores are high (>80%)
5. ✅ Verify Grad-CAM highlights correct areas

### Next Steps:
1. **Collect more data** for underrepresented class (Colletotrichum Blight)
2. **Monitor real-world performance**
3. **Gather user feedback**
4. **Retrain periodically** with new data
5. **Fine-tune** if accuracy needs improvement

### Performance Monitoring:
- Keep track of predictions
- Note any misclassifications
- Collect challenging cases for retraining
- Update model every 3-6 months with new data

---

## 🎓 Training Commands Quick Reference

```bash
# Full training workflow
cd backend

# 1. Verify setup
ls dataset/train/  # Should show 3 class folders

# 2. Start training
python train.py

# 3. Wait for training to complete (30-60 minutes with GPU)

# 4. Evaluate on test set
python evaluate.py

# 5. View training curves
open training_history.png  # or xdg-open on Linux

# 6. Check model file exists
ls -lh models/cardamom_model.pt

# 7. Restart backend with trained model
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 8. Test in browser
# Open http://localhost:8000/docs
# Upload a test image and check confidence!
```

---

## 📞 Need Help?

If you encounter issues:

1. **Check training logs** for error messages
2. **Verify dataset** organization and image quality
3. **Monitor GPU/CPU usage** during training
4. **Review training curves** for signs of overfitting/underfitting
5. **Test with known images** after deployment

---

## 🎉 Success Criteria

You'll know training was successful when:

- ✅ Training completes without errors
- ✅ Validation accuracy reaches 85%+ 
- ✅ Model file created: `models/cardamom_model.pt`
- ✅ Training curves show learning (not flat)
- ✅ API loads trained model successfully
- ✅ Real image predictions are accurate with high confidence (>80%)
- ✅ Grad-CAM highlights disease-affected areas

---

## 🚀 Ready to Train?

You have everything you need! Just run:

```bash
cd backend
python train.py
```

Good luck! Your model will be ready in about 30-60 minutes! 🎯
