# 🔄 What to Do After Pulling Changes

You've just pulled the latest changes from the repository. Here's your step-by-step guide to get everything working!

---

## 📋 Quick Checklist

```
┌─────────────────────────────────────────┐
│  Step 1: Install Dependencies          │
│  Step 2: Verify Installation            │
│  Step 3: Choose Your Path               │
│  Step 4: Start Using the System         │
└─────────────────────────────────────────┘
```

---

## Step 1: Install Dependencies 📦

### Backend (Required for all users)

```bash
cd backend
pip install -r requirements.txt
```

**Expected output**: All packages install successfully, no errors.

### Frontend (Optional - for web interface)

```bash
cd frontend
npm install
```

**Expected output**: Dependencies installed, no critical errors.

### Mobile App (Optional - for mobile development)

```bash
cd cardamom-mobile-app
npm install
```

**Expected output**: Dependencies installed, may see some warnings (normal).

---

## Step 2: Verify Installation ✅

### Check Backend

```bash
cd backend
python -c "import torch; import fastapi; print('✅ Backend dependencies OK')"
```

**Expected**: `✅ Backend dependencies OK`

### Check if you have training data

```bash
cd backend
ls -la dataset/
```

**What you might see:**
- ✅ If you see `train/`, `val/`, `test/` folders → **You have data!**
- ❌ If folder doesn't exist or is empty → **You need data first**

### Check if model is trained

```bash
cd backend
ls -la models/cardamom_model.pt
```

**What you might see:**
- ✅ File exists (~48 MB) → **Model is trained!**
- ❌ File not found → **Model needs training**

---

## Step 3: Choose Your Path 🎯

Based on your verification results, follow the appropriate path:

### Path A: You DON'T have training data yet ❌

**Status**: Need to collect data first

**Next steps:**
1. Read **[QUICK_START.md](QUICK_START.md)** - Decide how to get data
2. Read **[NEXT_STEPS.md](NEXT_STEPS.md)** - Complete roadmap

**Timeline**: 2-4 weeks for data collection

---

### Path B: You HAVE data but NO trained model ✅❌

**Status**: Ready to train!

**Next steps:**
1. Read **[YOURE_READY.md](YOURE_READY.md)** - You're ready to train! 🎉
2. Read **[START_TRAINING_NOW.md](START_TRAINING_NOW.md)** - Quick 3-step guide
3. Or read **[TRAINING_YOUR_MODEL.md](TRAINING_YOUR_MODEL.md)** - Detailed guide

**Quick start training:**
```bash
cd backend
python train.py
```

**Timeline**: 30-60 minutes (GPU) or 3-6 hours (CPU)

---

### Path C: You HAVE a trained model ✅✅

**Status**: Ready to use!

**Start the system:**

#### Option 1: Web Interface

Terminal 1 (Backend):
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

**Access**: Open http://localhost:5173 in your browser

#### Option 2: Mobile App

Terminal 1 (Backend):
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 (Mobile):
```bash
cd cardamom-mobile-app
npm start
```

**Important**: Update the API URL in `cardamom-mobile-app/src/services/api.ts` with your computer's IP address instead of localhost for phone testing.

---

## Step 4: Test the System 🧪

### Test Backend API

```bash
# In a new terminal
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "ok",
  "model_status": "trained"  // or "untrained"
}
```

### Test with a sample image

Visit: http://localhost:8000/docs

You'll see the interactive API documentation. Try the `/predict` endpoint with an image!

---

## 🔧 Troubleshooting

### "Module not found" errors

**Solution**: Reinstall dependencies
```bash
cd backend  # or frontend, or cardamom-mobile-app
pip install -r requirements.txt  # for backend
npm install  # for frontend/mobile
```

### "Port already in use" errors

**Solution**: Use a different port or kill existing process
```bash
# Backend
uvicorn app.main:app --reload --port 8001

# Frontend (in vite.config.ts, change port)
npm run dev -- --port 5174
```

### Backend starts but predictions are random (low confidence)

**This is expected!** The model needs training first.

- **Current**: ~35% confidence, random predictions
- **After training**: 90%+ confidence, accurate predictions

**Solution**: Follow **[START_TRAINING_NOW.md](START_TRAINING_NOW.md)** to train your model.

### Mobile app can't connect to backend

**Problem**: Phone can't reach "localhost"

**Solution**:
1. Find your computer's IP address:
   ```bash
   # On Linux/Mac
   ifconfig | grep "inet "
   
   # On Windows
   ipconfig
   ```

2. Update `cardamom-mobile-app/src/services/api.ts`:
   ```typescript
   const API_BASE_URL = 'http://192.168.1.100:8000';  // Use your IP
   ```

3. Make sure phone and computer are on same WiFi network

---

## 📚 Additional Documentation

### If You're New Here
- **[README.md](README.md)** - Project overview
- **[COMPLETE_SYSTEM_OVERVIEW.md](COMPLETE_SYSTEM_OVERVIEW.md)** - Full system documentation

### If You Have Data
- **[YOURE_READY.md](YOURE_READY.md)** - Celebration + ready check
- **[START_TRAINING_NOW.md](START_TRAINING_NOW.md)** - Quick training guide
- **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** - Training workflow overview

### If You Need Data
- **[QUICK_START.md](QUICK_START.md)** - Decision guide
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Complete roadmap

### For Issues
- **[ACCURACY_ISSUE_EXPLAINED.md](ACCURACY_ISSUE_EXPLAINED.md)** - Why low accuracy?
- **[WARNING_SYSTEM_GUIDE.md](WARNING_SYSTEM_GUIDE.md)** - Understanding warnings

### Reference
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation index
- **[MODEL_TRAINING.md](MODEL_TRAINING.md)** - Detailed training reference

---

## 🎯 Quick Decision Tree

```
After Pull → Install Dependencies → Verify Installation
                                          ↓
                          ┌───────────────┴───────────────┐
                          ↓                               ↓
                    Have Data?                      No Data?
                          ↓                               ↓
                    ┌─────┴─────┐                  Read QUICK_START.md
                    ↓           ↓                  Read NEXT_STEPS.md
              Have Model?   No Model?              Start collecting data
                    ↓           ↓                        (2-4 weeks)
              Start System  Train Model
              (Web/Mobile)  (30-60 min)
                    ↓           ↓
               Test & Use   Evaluate & Deploy
                    ↓           ↓
                    └───────────┘
                         ↓
                 Production Ready! 🎉
```

---

## ⚡ Common Scenarios

### Scenario 1: "I just want to see the interface"

**Do this:**
```bash
# Terminal 1
cd backend && uvicorn app.main:app --reload

# Terminal 2
cd frontend && npm run dev
```

Then open: http://localhost:5173

**Note**: Predictions will be random (untrained model) but UI will work!

---

### Scenario 2: "I have images and want accurate predictions"

**Do this:**
```bash
# First, organize your data (if not done)
cd backend
# Put images in dataset/colletotrichum_blight/, dataset/phyllosticta_leaf_spot/, dataset/healthy/

# Run split script
python split_dataset.py

# Train the model
python train.py

# Start system
uvicorn app.main:app --reload
```

**Timeline**: 30-60 minutes total

---

### Scenario 3: "I want to test on my phone"

**Do this:**
```bash
# Terminal 1: Start backend on all interfaces
cd backend
uvicorn app.main:app --reload --host 0.0.0.0

# Terminal 2: Start mobile app
cd cardamom-mobile-app
# First update API URL with your IP in src/services/api.ts
npm start
```

Then scan QR code with Expo Go app on your phone.

---

## ✅ Success Criteria

You've successfully set up the system when:

- [x] Backend starts without errors (`uvicorn app.main:app`)
- [x] Frontend loads in browser (http://localhost:5173)
- [x] API documentation accessible (http://localhost:8000/docs)
- [x] Health endpoint returns OK (`curl http://localhost:8000/health`)
- [x] Can upload image and get prediction (even if random/untrained)

For production accuracy, also need:
- [x] Training data collected (1,500+ images)
- [x] Model trained (`python train.py` completed)
- [x] Validation accuracy ≥85%
- [x] Predictions show 90%+ confidence

---

## 🆘 Need Help?

### Quick Help

**Dependencies won't install**: Check Python version (need 3.8+) and Node version (need 14+)

**Backend won't start**: Check if port 8000 is free, or use different port

**Frontend won't start**: Check if port 5173 is free, run `npm install` again

**Low accuracy**: Expected with untrained model. Train it!

**Can't find model file**: Model doesn't exist yet. Train it with `python train.py`

### Documentation Reference

Check **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** for complete guide to all documentation.

---

## 📝 Summary

**After pulling changes:**

1. ✅ Install dependencies (backend, frontend, mobile)
2. ✅ Verify what you have (data? model?)
3. ✅ Choose your path:
   - No data → Collect data
   - Have data, no model → Train model
   - Have model → Start using system
4. ✅ Test and verify everything works

**Most common next step**: If you have data, train the model with `python train.py`

**Questions?** Check the documentation or open an issue!

---

**🎉 Welcome back! You're ready to continue your cardamom disease detection journey!**
