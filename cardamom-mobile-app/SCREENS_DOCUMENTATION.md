# Mobile App Screen Flows and Features

## Complete User Journey

### 1. Home Screen (होम स्क्रीन)

**Layout:**
```
┌─────────────────────────────────────┐
│  🌿 अलैंची रोग पहिचान              │
│  Cardamom Disease Detection         │
│                                     │
│  अलैंची बिरुवाको पातको तस्बिर      │
│  खिच्नुहोस् वा ग्यालेरीबाट छान्नुहोस्│
│                                     │
│  ┌───────────────────────────────┐  │
│  │       📷                      │  │
│  │  क्यामेरा खोल्नुहोस्          │  │
│  │  Open Camera                  │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │       🖼️                      │  │
│  │  ग्यालेरीबाट छान्नुहोस्       │  │
│  │  Choose from Gallery          │  │
│  └───────────────────────────────┘  │
│                                     │
│  📋 समर्थित रोगहरू:                 │
│  • कोलेटोट्रिकम ब्लाइट            │
│  • फाइलोस्टिक्टा पात दाग          │
│  • स्वस्थ                          │
│                                     │
│  💡 राम्रो नतिजाको लागि:           │
│  • स्पष्ट र फोकसमा रहेको तस्बिर     │
│  • राम्रो प्रकाशमा तस्बिर           │
│  • पातको नजिकबाट तस्बिर            │
│  • पूरै पात फ्रेममा देखिने गरी     │
└─────────────────────────────────────┘
```

**Features:**
- Bilingual interface (English/Nepali)
- Two main actions: Camera and Gallery
- Disease list preview
- Photography tips in Nepali
- Gradient purple/blue background
- Large, touch-friendly buttons

---

### 2. Image Capture/Selection

**Camera Mode:**
- Native camera integration via expo-camera
- Real-time preview
- 1:1 aspect ratio for consistent processing
- Capture button
- Cancel option to return to home

**Gallery Mode:**
- Access device photo library via expo-image-picker
- Image cropping enabled (1:1 aspect)
- Quality setting: 80%
- Multi-format support (JPEG, PNG)

---

### 3. Processing State

**During API Call:**
```
┌─────────────────────────────────────┐
│                                     │
│         ⏳ (spinning)                │
│                                     │
│    तस्बिर विश्लेषण गर्दै...         │
│    (Analyzing image...)             │
│                                     │
└─────────────────────────────────────┘
```

**What Happens:**
1. Image uploaded to FastAPI backend
2. Backend performs preprocessing
3. CNN model inference
4. Grad-CAM heatmap generation
5. Results returned to app

---

### 4. Result Screen (नतिजा स्क्रीन)

**Layout:**
```
┌─────────────────────────────────────┐
│  ← परिणाम                           │
│                                     │
│  तस्बिर (Original Image)            │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │     [Uploaded Image]          │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ रोगको नाम (Disease)            │  │
│  │                               │  │
│  │ Colletotrichum Blight         │  │
│  │ कोलेटोट्रिकम ब्लाइट           │  │
│  │                               │  │
│  │ विश्वास स्तर (Confidence)      │  │
│  │ 85.43% ████████████░░         │  │
│  └───────────────────────────────┘  │
│                                     │
│  ग्रेड-CAM हिटम्याप (Grad-CAM)      │
│  यो दृश्यले देखाउँछ कि कुन क्षेत्र... │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │  [Heatmap Overlay Image]      │  │
│  │  (Red/Orange highlights)      │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│  रोग जानकारी                       │
│  ┌───────────────────────────────┐  │
│  │ Colletotrichum Blight  [High] │  │
│  │ कोलेटोट्रिकम ब्लाइट           │  │
│  │                               │  │
│  │ यो रोग कोलेटोट्रिकम प्रजाति... │  │
│  │                               │  │
│  │ थप जानकारी हेर्नुहोस् →        │  │
│  └───────────────────────────────┘  │
│                                     │
│  [ℹ️ विस्तृत जानकारी]              │
│  [📷 नयाँ तस्बिर]                  │
└─────────────────────────────────────┘
```

**Features:**
- Original image display
- Disease name in both languages
- Confidence score with visual progress bar
- Color-coded confidence (Green >80%, Orange 60-80%, Red <60%)
- Warning for low confidence predictions
- Grad-CAM heatmap with Nepali explanation
- Quick disease info card
- Action buttons: View Details / New Photo

---

### 5. Disease Info Screen (रोग जानकारी स्क्रीन)

**Layout:**
```
┌─────────────────────────────────────┐
│  ← Colletotrichum Blight            │
│     कोलेटोट्रिकम ब्लाइट             │
│                                     │
│  [अत्यन्त गम्भीर] (Red badge)        │
│                                     │
│  📄 विवरण                           │
│  कोलेटोट्रिकम ब्लाइट अलैंची बालीमा...│
│  (Full description in Nepali)       │
│                                     │
│  🏥 लक्षणहरू                         │
│  • पातहरूमा गहिरो खैरो वा कालो दाग...│
│  • पातको किनारामा सुक्खापन र झर्नु...│
│  • दागहरू बढ्दै गएर सम्पूर्ण पातमा...│
│  • बिरुवाको पातहरू पहेंलो हुनु...   │
│  (6 symptoms listed)                │
│                                     │
│  📊 कारण                            │
│  यो रोग अत्यधिक आर्द्रता, खराब हावा...│
│  (Detailed causes explanation)      │
│                                     │
│  💊 उपचार तथा व्यवस्थापन              │
│  1. संक्रमित पातहरू र बिरुवाका...   │
│  2. फंगीसाइड जस्तै कपर आधारित...   │
│  3. Mancozeb (२.५ ग्राम प्रति...)   │
│  (6 treatment steps)                │
│                                     │
│  🛡️ रोकथाम                          │
│  ✓ बिरुवाहरू बीचमा उचित दूरी...     │
│  ✓ राम्रो निकास र हावा संचारको...   │
│  ✓ बिरुवालाई बिहान पानी दिनुहोस्... │
│  (7 prevention tips)                │
│                                     │
│  ⏰ कहिले कारबाही गर्ने               │
│  रोगको प्रारम्भिक लक्षण देखिएपछि...  │
│  (Action timeline guidance)         │
│                                     │
│  [🏠 मुख्य पृष्ठमा फर्कनुहोस्]       │
└─────────────────────────────────────┘
```

**Features:**
- Full-screen disease information
- Color-coded severity badge
- Organized sections with icons:
  - Description (विवरण)
  - Symptoms (लक्षणहरू)
  - Causes (कारण)
  - Treatment (उपचार)
  - Prevention (रोकथाम)
  - When to Act (कहिले कारबाही गर्ने)
- All text in Nepali
- Scrollable content
- Back to home button

---

## Technical Features

### API Integration
- **Endpoint**: POST http://localhost:8000/predict
- **Upload**: FormData with image file
- **Response**: JSON with class_name, confidence, heatmap (base64)
- **Timeout**: 30 seconds
- **Error Handling**: Network errors, server errors, timeouts

### Data Structure
```typescript
interface DiseaseInfo {
  id: string;
  nameEnglish: string;
  nameNepali: string;
  descriptionNepali: string;
  symptomsNepali: string[];
  causesNepali: string;
  treatmentNepali: string[];
  preventionNepali: string[];
  whenToActNepali: string;
  severity: 'low' | 'medium' | 'high';
}
```

### Navigation Flow
```
Home → Camera/Gallery → [API Call] → Result → Disease Info
  ↑                                      ↓           ↓
  └──────────────────────────────────────┴───────────┘
               (Back navigation)
```

### Color Scheme
- **Primary**: #667eea (Purple)
- **Secondary**: #764ba2 (Dark Purple)
- **Success**: #4caf50 (Green)
- **Warning**: #ff9800 (Orange)
- **Error**: #f44336 (Red)
- **Background**: #f5f7fa (Light Gray)

### Permissions Required
- **Camera**: For taking photos
- **Photo Library**: For selecting existing images
- **Internet**: For API communication

---

## Comprehensive Disease Database

### Disease 1: Colletotrichum Blight
- **Severity**: High (अत्यन्त गम्भीर)
- **Symptoms**: 6 detailed points
- **Treatment**: 6 step-by-step instructions with product names and dosages
- **Prevention**: 7 preventive measures
- **Total Content**: ~2,000 characters in Nepali

### Disease 2: Phyllosticta Leaf Spot
- **Severity**: Medium (मध्यम गम्भीर)
- **Symptoms**: 6 detailed points
- **Treatment**: 6 step-by-step instructions
- **Prevention**: 7 preventive measures
- **Total Content**: ~1,800 characters in Nepali

### Disease 3: Healthy
- **Severity**: Low (कम गम्भीर)
- **Description**: Positive reinforcement
- **Symptoms**: 4 healthy indicators
- **Care Tips**: Maintenance recommendations
- **Total Content**: ~800 characters in Nepali

---

## User Experience Highlights

### For Farmers (किसानहरूको लागि)
1. **Simple Interface**: Large buttons, clear icons
2. **Nepali Language**: Complete information in native language
3. **Visual Guidance**: Tips for taking good photos
4. **Actionable Info**: Specific treatment instructions with dosages
5. **Prevention Focus**: Detailed preventive measures
6. **Offline Ready**: Disease info can be cached locally

### For Extension Workers
1. **Quick Assessment**: Fast disease identification
2. **Educational Tool**: Comprehensive disease information
3. **Visual Proof**: Grad-CAM shows model reasoning
4. **Shareable Results**: Can be used during field visits

### Technical Excellence
1. **Type Safety**: Full TypeScript implementation
2. **Modern Architecture**: React Navigation, component-based
3. **Performance**: Optimized image handling
4. **Error Resilience**: Comprehensive error handling
5. **Maintainable**: Clean code structure, documented

---

## Future Enhancements Possible

- **History Tracking**: Save past diagnoses
- **Offline Mode**: Local model inference
- **Multi-Image**: Batch processing
- **Location Services**: Regional disease alerts
- **Community Features**: Share with other farmers
- **Audio Guide**: Voice instructions in Nepali
- **Weather Integration**: Link disease risk to weather
- **Crop Management**: Full farm management features
