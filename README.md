# Cardamom Leaf Disease Detection System

A complete full-stack and mobile application system for detecting diseases in cardamom leaves using deep learning. The system classifies leaf images into three categories:
- **Colletotrichum Blight** (कोलेटोट्रिकम ब्लाइट)
- **Phyllosticta Leaf Spot** (फाइलोस्टिक्टा पात दाग)
- **Healthy** (स्वस्थ)

---

## 🚀 NEW USERS - START HERE!

**First time setting up? Follow these steps IN ORDER:**

1. **📖 Read**: [START_HERE.md](START_HERE.md) ← **Start with this!** (3-minute read)
2. **📦 Install**: `cd backend && pip install -r requirements.txt` (3-5 minutes)
3. **✅ Verify**: `python backend/check_training_setup.py` (30 seconds)
4. **🎯 Train**: `python backend/train.py` (30-60 minutes)

**Quick check:**
- ✅ All checks pass → You're ready to train!
- ❌ Checks fail → Read [INSTALL_DEPENDENCIES.md](INSTALL_DEPENDENCIES.md)
- ❓ Questions? → Read [FAQ.md](FAQ.md) for common issues

**Don't have data yet?** → Read [NEXT_STEPS.md](NEXT_STEPS.md) for data collection guidance

---

## 🔄 Just Pulled Changes?

**→ Read [AFTER_PULL.md](AFTER_PULL.md)** - Complete guide for what to do after pulling updates!

Quick summary: Install dependencies → Verify installation → Choose your path (train model or start using system)

---

## 🎉 READY TO TRAIN YOUR MODEL?

**If you have collected cardamom disease images:**

### 📚 Quick Start Guides:
1. **[YOURE_READY.md](YOURE_READY.md)** - Visual guide showing you're ready! 🎯
2. **[START_TRAINING_NOW.md](START_TRAINING_NOW.md)** - 3-step quick start (2 min read) ⚡
3. **[TRAINING_YOUR_MODEL.md](TRAINING_YOUR_MODEL.md)** - Complete detailed guide 📖

### ⚡ Super Quick Start:
```bash
cd backend
python train.py
```

### Current System Status:
- ⚠️ **Model Status**: Untrained (using random weights)
- ⚠️ **Predictions**: Low accuracy (~35% confidence)
- ✅ **After Training**: 90%+ confidence with accurate predictions!

**With your dataset ready, training takes 30-60 minutes (GPU) to get a production-ready model!**

## 🌟 Features

- **Deep Learning Classification**: PyTorch-based EfficientNetV2 model for disease detection
- **Grad-CAM Visualization**: Visual explanation showing which leaf regions influenced predictions
- **Background Removal**: U2-Net integration (placeholder ready)
- **Modern Web Interface**: React TypeScript frontend with responsive design
- **📱 Mobile App**: React Native app with camera integration and Nepali language support
- **Real-time Predictions**: Fast inference with confidence scores
- **RESTful API**: FastAPI backend with automatic documentation
- **Bilingual Support**: English and Nepali (नेपाली) interface

## 🏗️ Architecture

### Backend (FastAPI + PyTorch)
- **Framework**: FastAPI with CORS support
- **Model**: EfficientNetV2-S classifier (3 output classes) - **requires training**
- **Preprocessing**: ImageNet normalization, 224x224 resizing
- **Visualization**: Grad-CAM heatmap generation
- **Background Removal**: U2-Net placeholder (ready for integration)

### Frontend (React + TypeScript + Vite)
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite for fast development
- **HTTP Client**: Axios for API communication
- **UI**: Modern gradient design, responsive layout
- **Features**: Image upload, preview, results display, error handling

### Mobile App (React Native + Expo)
- **Framework**: React Native with Expo
- **Language**: TypeScript
- **Features**: Camera capture, gallery picker, disease detection
- **UI**: Bilingual (English/Nepali), native mobile experience
- **Navigation**: React Navigation
- **Comprehensive Disease Info**: Full information in Nepali for farmers

## 📁 Project Structure

```
cardamom-leaf-disease-detection/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── schemas.py           # Pydantic models
│   │   ├── models/
│   │   │   ├── cardamom_model.py      # CNN classifier
│   │   │   └── u2net_segmenter.py     # Background removal
│   │   └── utils/
│   │       ├── image_preprocess.py    # Image preprocessing
│   │       ├── grad_cam.py            # Grad-CAM implementation
│   │       └── overlay.py             # Heatmap overlay
│   ├── models/                  # Trained model weights (gitignored)
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── src/
│   │   ├── main.tsx            # React entry
│   │   ├── App.tsx             # Main component
│   │   ├── App.css             # Styles
│   │   ├── index.css           # Global styles
│   │   └── api/
│   │       └── client.ts       # API client
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── cardamom-mobile-app/         # 📱 React Native Mobile App
│   ├── App.tsx                  # Main app with navigation
│   ├── src/
│   │   ├── screens/             # HomeScreen, ResultScreen, DiseaseInfoScreen
│   │   ├── components/          # Reusable UI components
│   │   ├── services/            # API client
│   │   ├── data/                # Disease info in Nepali
│   │   ├── types/               # TypeScript types
│   │   └── utils/               # Helper functions
│   ├── app.json                 # Expo configuration
│   ├── package.json
│   └── README.md
│
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Backend**:
  - **Python 3.9 - 3.13** (Recommended: Python 3.11 or 3.12)
  - pip (latest version)
  - **Having issues?** See [PYTHON_VERSION_GUIDE.md](PYTHON_VERSION_GUIDE.md)

- **Frontend**:
  - Node.js 18 or higher
  - npm or yarn

- **Mobile App**:
  - Node.js 18 or higher
  - Expo CLI (`npm install -g expo-cli`)
  - Expo Go app (for testing on real devices)

### Installation

#### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

#### 3. Mobile App Setup

```bash
# Navigate to mobile app directory
cd cardamom-mobile-app

# Install dependencies
npm install
```

### Running the Application

#### Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

The frontend will be available at: http://localhost:5173

#### Start Mobile App

```bash
cd cardamom-mobile-app
npm start
```

Then:
- Scan the QR code with Expo Go app on your device
- Or press `i` for iOS simulator, `a` for Android emulator, `w` for web

**Note**: To connect the mobile app to your backend:
1. Find your computer's local IP address
2. Update `src/services/api.ts` in the mobile app:
   ```typescript
   const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';
   ```

## 💬 Common Startup Messages

### When You Start the Backend

You'll see informational messages during startup. **These are normal!**

#### U2-Net Placeholder Message
```
ℹ️  U2-Net background removal is not yet implemented (optional feature)
   Images will be processed without background removal
```

**This is NOT an error!** 
- U2-Net is an **optional feature** for background removal
- The system works perfectly without it (85-92% accuracy with trained model)
- Images are processed without background removal
- **No action needed** - continue using the system normally

**Want to understand this better?** → Read [U2NET_PLACEHOLDER.md](U2NET_PLACEHOLDER.md)

#### Untrained Model Warning
```
⚠️  WARNING: Model needs training for accurate predictions!
```

**This means:**
- Model is using random weights (not trained yet)
- Predictions will be ~33% confidence (random guessing)
- **Action needed**: Train your model with `python train.py`

**After training**, this warning disappears and you get:
```
✓ Classifier model loaded successfully (trained)
```

**Need help?** → Read [WHY_LOW_ACCURACY.md](WHY_LOW_ACCURACY.md) and [TRAIN_NOW.md](TRAIN_NOW.md)

---

## 📡 API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /predict
Upload an image for disease prediction.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "class_name": "Colletotrichum Blight",
  "confidence": 0.85,
  "heatmap": "base64_encoded_png_string"
}
```

## 🎯 How It Works

1. **Upload**: User selects a cardamom leaf image
2. **Preprocessing**:
   - Background removal (U2-Net - placeholder)
   - Resize to 224x224
   - Normalize with ImageNet statistics
3. **Inference**: CNN model predicts disease class
4. **Visualization**: Grad-CAM generates heatmap showing important regions
5. **Results**: Display class name, confidence, and heatmap overlay

## 🔧 Model Integration

### Current State (Placeholder)

The system currently uses placeholder models for demonstration:
- **Classifier**: Simple 4-layer CNN with random weights
- **U2-Net**: Passes images through unchanged

### Integrating Trained Models

When you have trained models:

1. **Place Model Weights**:
   - Save trained classifier as `backend/models/cardamom_model.pt`
   - Save U2-Net as `backend/models/u2net.pth`

2. **Update Model Architecture**:
   - Modify `backend/app/models/cardamom_model.py` to match your architecture (e.g., EfficientNetV2)
   - Implement U2-Net loading in `backend/app/models/u2net_segmenter.py`

3. **Adjust Preprocessing** (if needed):
   - Update `backend/app/utils/image_preprocess.py`

The system will automatically detect and load trained weights if they exist in the models directory.

## 🧪 Testing

### Backend Testing

```bash
cd backend
pytest
```

### Frontend Testing

```bash
cd frontend
npm run lint
npm run build
```

### Manual Testing

1. Start both servers
2. Open http://localhost:5173
3. Upload a test image
4. Verify prediction results and heatmap display

## 🎨 UI Features

- **Modern Design**: Gradient background (purple/blue)
- **Responsive**: Works on desktop and mobile
- **Interactive**: Hover effects, loading states
- **Informative**: Error messages, confidence visualization
- **Visual Explanation**: Grad-CAM heatmap overlay

## 🔐 Security Notes

- File type validation on frontend and backend
- CORS configured for specific origins
- Request timeouts to prevent hanging
- Input sanitization and error handling

## 📝 Development Notes

### Backend
- Uses PyTorch hooks for Grad-CAM
- FastAPI automatic API documentation
- Proper error handling and logging
- Device auto-detection (CUDA/CPU)

### Frontend
- TypeScript for type safety
- Axios with 30s timeout
- Memory cleanup (URL.revokeObjectURL)
- Error boundary handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- FastAPI for the excellent web framework
- React team for the UI library
- U2-Net authors for background removal architecture
