# Cardamom Disease Detection Mobile App

अलैंची रोग पहिचान मोबाइल एप

A React Native mobile application for detecting diseases in cardamom leaves using AI. The app integrates with the FastAPI backend to provide real-time disease detection with Nepali language support.

## Features

### Core Functionality
- 📷 **Camera Integration**: Capture leaf images directly from the app
- 🖼️ **Gallery Picker**: Select existing images from your device
- 🤖 **AI-Powered Detection**: Real-time disease prediction using FastAPI backend
- 🔥 **Grad-CAM Visualization**: See which parts of the leaf influenced the prediction
- 🌐 **Bilingual Support**: English and Nepali (नेपाली) interface

### Disease Information
Complete information in Nepali for 3 disease classes:
1. **Colletotrichum Blight** (कोलेटोट्रिकम ब्लाइट)
2. **Phyllosticta Leaf Spot** (फाइलोस्टिक्टा पात दाग)
3. **Healthy** (स्वस्थ)

Each disease includes:
- Detailed description (विवरण)
- Symptoms (लक्षणहरू)
- Causes (कारण)
- Treatment recommendations (उपचार तथा व्यवस्थापन)
- Prevention tips (रोकथाम)
- When to take action (कहिले कारबाही गर्ने)

## Tech Stack

- **Framework**: React Native with Expo
- **Language**: TypeScript
- **Navigation**: React Navigation
- **Image Handling**: 
  - expo-camera (Camera capture)
  - expo-image-picker (Gallery selection)
- **API Communication**: Axios
- **UI Components**: React Native Paper
- **Icons**: @expo/vector-icons

## Project Structure

```
cardamom-mobile-app/
├── App.tsx                      # Main app component with navigation
├── src/
│   ├── screens/
│   │   ├── HomeScreen.tsx       # Main screen with camera/gallery options
│   │   ├── ResultScreen.tsx     # Display prediction results
│   │   └── DiseaseInfoScreen.tsx # Detailed disease information
│   ├── components/
│   │   ├── ImagePreview.tsx     # Image display component
│   │   ├── DiseaseCard.tsx      # Disease information card
│   │   ├── HeatmapViewer.tsx    # Grad-CAM heatmap viewer
│   │   └── LoadingSpinner.tsx   # Loading indicator
│   ├── services/
│   │   └── api.ts               # API client for backend communication
│   ├── data/
│   │   └── diseaseInfo.ts       # Nepali disease information database
│   ├── types/
│   │   └── index.ts             # TypeScript type definitions
│   └── utils/
│       └── imageHelper.ts       # Image processing utilities
├── assets/                      # Images, icons, fonts
├── app.json                     # Expo configuration
├── package.json                 # Dependencies
└── README.md                    # This file
```

## Prerequisites

- Node.js 18 or higher
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- **Backend server running on http://localhost:8000**

For iOS development:
- macOS with Xcode (or use Expo Go app)

For Android development:
- Android Studio with Android SDK (or use Expo Go app)

## Installation

1. **Navigate to the mobile app directory:**
```bash
cd cardamom-mobile-app
```

2. **Install dependencies:**
```bash
npm install
```

3. **Ensure backend is running:**
Make sure the FastAPI backend is running on http://localhost:8000. See the backend README for setup instructions.

## Running the App

### Option 1: Using Expo Go (Recommended for Development)

1. **Start the Expo development server:**
```bash
npm start
```

2. **Scan the QR code:**
   - iOS: Use the Camera app to scan the QR code
   - Android: Use the Expo Go app to scan the QR code

3. **Note**: To connect to the backend on your local machine:
   - Find your computer's local IP address
   - Update the API_BASE_URL in `src/services/api.ts`:
   ```typescript
   const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';
   ```

### Option 2: iOS Simulator (macOS only)

```bash
npm run ios
```

### Option 3: Android Emulator

```bash
npm run android
```

### Option 4: Web Browser (Limited Functionality)

```bash
npm run web
```

**Note**: Camera functionality may be limited in web browsers.

## Backend Configuration

The app expects the backend API to be running at `http://localhost:8000` by default.

To change the backend URL, edit `src/services/api.ts`:

```typescript
const API_BASE_URL = 'http://YOUR_BACKEND_URL:PORT';
```

### For Development on Real Devices:

1. Find your computer's local IP address:
   - **Windows**: `ipconfig` (look for IPv4 Address)
   - **macOS/Linux**: `ifconfig` or `ip addr` (look for inet address)

2. Update the API URL:
```typescript
const API_BASE_URL = 'http://192.168.1.XXX:8000'; // Replace with your IP
```

3. Ensure the backend allows connections from your device's IP (CORS configuration).

## API Endpoints Used

### Health Check
- **GET** `/health`
- Returns server status

### Predict Disease
- **POST** `/predict`
- Accepts: `multipart/form-data` with image file
- Returns:
  ```json
  {
    "class_name": "Colletotrichum Blight",
    "confidence": 0.85,
    "heatmap": "base64_encoded_png_string"
  }
  ```

## Features in Detail

### Home Screen
- Two main options: Camera and Gallery
- Lists supported disease classes
- Provides tips for better results
- Nepali and English labels

### Result Screen
- Displays original image
- Shows predicted disease name (English and Nepali)
- Confidence score with visual bar
- Grad-CAM heatmap overlay
- Quick access to detailed disease info

### Disease Info Screen
- Comprehensive disease information in Nepali
- Severity indicator
- Symptoms list
- Causes explanation
- Treatment recommendations
- Prevention tips
- Action timeline

## Troubleshooting

### Cannot Connect to Backend

**Error**: "Cannot connect to server"

**Solutions**:
1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check if using correct IP address (not localhost on real device)
3. Verify firewall settings allow connections
4. Check CORS configuration in backend

### Camera/Gallery Not Working

**Solutions**:
1. Grant camera and photo library permissions in device settings
2. Restart the Expo development server
3. Clear Expo cache: `expo start -c`

### Bundling Error: "Unable to resolve @babel/runtime"

**Error Message**:
```
Unable to resolve "./unsupportedIterableToArray.js" from "node_modules/@babel/runtime/helpers/slicedToArray.js"
```

**Solutions**:
1. The `@babel/runtime` package is now included in dependencies
2. If you still see this error, try:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   npm start
   ```
3. Clear Metro bundler cache:
   ```bash
   npm start -- --clear
   # or
   expo start -c
   ```

### Low Prediction Confidence

**Solutions**:
1. Ensure good lighting conditions
2. Focus properly on the leaf
3. Capture clear, high-resolution images
4. Include the entire leaf in the frame
5. Avoid blurry or dark images

## Building for Production

### iOS App Store

```bash
expo build:ios
```

### Android Play Store

```bash
expo build:android
```

For detailed build instructions, see [Expo Build Documentation](https://docs.expo.dev/build/introduction/).

## Development

### Running in Development Mode

```bash
npm start
```

### Type Checking

```bash
npx tsc --noEmit
```

### Linting

```bash
npm run lint
```

## Future Enhancements

- [ ] Offline mode with cached predictions
- [ ] History of analyzed images
- [ ] Share results via social media
- [ ] Multiple image batch processing
- [ ] Location-based disease alerts
- [ ] Push notifications for disease outbreaks
- [ ] Audio guidance in Nepali
- [ ] Integration with farmer advisory systems

## License

MIT

## Credits

Developed for cardamom farmers in Nepal to help identify and manage leaf diseases.

## Support

For issues or questions:
1. Check the [backend README](../backend/README.md) for API issues
2. Ensure you're using compatible versions
3. Review the troubleshooting section above

---

**Note**: This app requires an active internet connection and the FastAPI backend running to perform disease detection.
