# 🚀 START HERE - First Time Setup

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     █████  ████████  █████  ████████  ████████              ║
║    ██   ██    ██    ██   ██    ██        ██                  ║
║    ███████    ██    ███████    ██        ██                  ║
║    ██   ██    ██    ██   ██    ██        ██                  ║
║                                                               ║
║    ██   ██ ██████  ████████  ██████                          ║
║    ██   ██ ██      ██   ██  ██                               ║
║    ███████ ██████  ████████  ██████                          ║
║    ██   ██ ██      ██   ██       ██                          ║
║                                                               ║
║            FIRST TIME SETUP GUIDE                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## Welcome! 👋

This is your **first-time setup guide** for the Cardamom Leaf Disease Detection System.

Follow these **4 simple steps** to get everything working:

---

## 💡 STEP 0: Virtual Environment (RECOMMENDED)

**What:** Create an isolated Python environment  
**Time:** 1 minute  
**Why:** Prevents conflicts, keeps project clean

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it - IMPORTANT!
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

**How to tell it's active:** Your prompt should show `(venv)`:
```bash
(venv) user@computer:~/backend$
```

**🎉 User Discovery:** Someone forgot this step and had issues. After activating venv, everything worked perfectly!

> "i forgot to do this: source venv/bin/activate  
> Everything is working fine now"

**Learn more:** See **VIRTUAL_ENVIRONMENT.md** for complete guide.

---

## ⚠️ IMPORTANT: Git Pull vs Pip Install

```
┌─────────────────────────────────────────┐
│  git pull    → Gets code & docs          │
│  pip install → Installs packages         │
│                                          │
│  BOTH are needed!                       │
└─────────────────────────────────────────┘
```

**Running `git pull` does NOT install Python packages!**  
After pulling changes, you must run: `pip install -r requirements.txt`

See **FAQ.md** for more details.

---

## ⚡ STEP 1: Install Dependencies

**What:** Install all required Python packages  
**Time:** 3-5 minutes  
**Size:** ~160MB download

```bash
# Make sure venv is active first! (See STEP 0)
cd backend
pip install -r requirements.txt
```

**What gets installed:**
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision models  
- **Pillow** - Image processing
- **opencv-python** - Computer vision operations
- **NumPy** - Numerical computing
- **tqdm** - Progress bars
- **matplotlib** - Visualization
- **fastapi, uvicorn, pydantic** - API framework
- **python-multipart** - File uploads

**Expected output:** "Successfully installed..." messages for all packages

---

## ✅ STEP 2: Verify Installation

**What:** Check that everything is installed correctly  
**Time:** 30 seconds

```bash
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
✅ Pillow               - OK

✅ ALL CHECKS PASSED!

🚀 You're ready to train!
```

---

## 🎯 STEP 3: Start Training

**What:** Train the model on your cardamom disease dataset  
**Time:** 30-60 minutes (depends on your computer)

```bash
python train.py
```

**Expected output:** Progress bars showing training progress

```
Epoch 1/50
Training: 100%|████████| 38/38 [00:45<00:00]
Validation: 100%|████████| 8/8 [00:05<00:00]
Epoch 1/50 - Train Loss: 0.8234, Acc: 67.2% | Val Loss: 0.6123, Acc: 75.3%
```

---

## ❌ What If Something Fails?

### If STEP 1 fails (installation error):
- **Python too old?** Need Python 3.9-3.13
  - Check: `python --version`
  - Upgrade if needed
- **Permission error?** Add `--user` flag
  - `pip install --user -r requirements.txt`
- **Network timeout?** Try again or increase timeout
  - `pip install --timeout=300 -r requirements.txt`
- **More help:** Read `INSTALL_DEPENDENCIES.md`

### If STEP 2 fails (checks don't pass):
- **Imports missing?** Run STEP 1 again
- **Dataset missing?** See `NEXT_STEPS.md` for data collection
- **Device issues?** Check `PYTHON_VERSION_GUIDE.md`
- **More help:** Read `TRAIN_PY_ERRORS.md`

### If STEP 3 fails (training error):
- **Still missing packages?** Run STEP 1 and 2 again
- **Dataset error?** Verify your dataset is organized correctly
- **Out of memory?** Reduce batch size in train.py
- **More help:** Read `TRAIN_ERROR_SOLUTION.md`

---

## 📚 Next Steps After Training

Once training completes successfully:

1. **Evaluate your model:**
   ```bash
   python evaluate.py
   ```

2. **Start the API backend:**
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Try the web interface:**
   ```bash
   cd ../frontend
   npm install
   npm run dev
   ```

4. **Try the mobile app:**
   ```bash
   cd ../cardamom-mobile-app
   npm install
   npm start
   ```

---

## 🆘 Need More Help?

### Quick References
- **Installation problems** → `INSTALL_DEPENDENCIES.md`
- **Training errors** → `TRAIN_PY_ERRORS.md`
- **Python version issues** → `PYTHON_VERSION_GUIDE.md`
- **What to do next** → `NEXT_STEPS.md`
- **All documentation** → `DOCUMENTATION_INDEX.md`

### Complete Guides
- **Quick training guide** → `START_TRAINING_NOW.md`
- **Detailed training guide** → `TRAINING_YOUR_MODEL.md`
- **After pulling changes** → `AFTER_PULL.md`

---

## ✅ Success Checklist

Before you start training, make sure:

- [ ] Python 3.9-3.13 is installed
- [ ] You're in the `backend/` directory
- [ ] `pip install -r requirements.txt` completed successfully
- [ ] `python check_training_setup.py` shows all ✅
- [ ] You have your dataset organized in `dataset/` folder

If all boxes are checked, you're ready to go! 🚀

```bash
python train.py
```

---

## 💡 Pro Tips

1. **Always run the diagnostic first:** `python check_training_setup.py`
2. **Use the automated setup script:** `./setup.sh` (does steps 1-2 automatically)
3. **Read error messages carefully:** They tell you exactly how to fix issues
4. **Check the documentation:** We have comprehensive guides for everything

---

**Happy Training! 🎉**

Your model will learn to detect three conditions:
- Colletotrichum Blight (कोलेटोट्रिकम ब्लाइट)
- Phyllosticta Leaf Spot (फाइलोस्टिक्टा पात दाग) 
- Healthy (स्वस्थ)

Good luck with your cardamom disease detection system! 🌱
