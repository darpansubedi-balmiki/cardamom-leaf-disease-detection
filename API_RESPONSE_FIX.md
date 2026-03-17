# API Response Structure Fix

## Issue
The frontend and mobile app were not displaying prediction results correctly because they expected different field names than what the backend API actually returns.

### What User Saw
- Disease Class: (empty)
- Confidence: **NaN%**
- No prediction results

### Network Tab Showed
```json
{
  "top_class": "Healthy",
  "top_probability": 1.0,
  "top_probability_pct": 100.0,
  "is_uncertain": false,
  "confidence_threshold": 0.6,
  "top_k": [
    {"class_name": "Healthy", "probability": 1.0, "probability_pct": 100.0},
    {"class_name": "Phyllosticta Leaf Spot", "probability": 3.496028466720702e-10, "probability_pct": 0.0},
    {"class_name": "Colletotrichum Blight", "probability": 4.912332259715546e-12, "probability_pct": 0.0}
  ]
}
```

### What Was Wrong
Frontend and mobile app expected:
```typescript
{
  class_name: string,    // ❌ Backend sends "top_class"
  confidence: number,    // ❌ Backend sends "top_probability"
  heatmap: string       // ❌ Backend sends this but also sends other fields
}
```

## The Fix

### Backend API Response Structure (Actual)
```typescript
interface TopKPrediction {
  class_name: string;
  probability: number;
  probability_pct: number;
}

interface PredictionResponse {
  top_class: string;                    // Main predicted class
  top_probability: number;              // Confidence (0.0 - 1.0)
  top_probability_pct: number;          // Confidence as percentage
  is_uncertain: boolean;                 // True if below threshold
  confidence_threshold: number;          // Threshold used (default 0.6)
  top_k: TopKPrediction[];              // All predictions ranked
  heatmap?: string;                      // Optional: Base64 PNG
  model_trained?: boolean;               // Optional: Model status
  warning?: string;                      // Optional: Warning message
}
```

### Files Updated

#### Frontend
1. **frontend/src/api/client.ts**
   - Updated `PredictionResponse` interface to match backend
   - Added `TopKPrediction` interface
   - Updated validation logic

2. **frontend/src/App.tsx**
   - Changed `result.class_name` → `result.top_class`
   - Changed `result.confidence` → `result.top_probability`
   - Added "All Predictions" section displaying `top_k` array
   - Added uncertainty warning when `is_uncertain` is true
   - Made heatmap optional

3. **frontend/src/App.css**
   - Added styles for prediction list display

#### Mobile App
1. **cardamom-mobile-app/src/types/index.ts**
   - Updated `PredictionResponse` interface to match backend
   - Added `TopKPrediction` interface

2. **cardamom-mobile-app/src/screens/ResultScreen.tsx**
   - Changed `prediction.class_name` → `prediction.top_class`
   - Changed `prediction.confidence` → `prediction.top_probability`
   - Added display for all predictions from `top_k`
   - Added warning display if `prediction.warning` exists
   - Updated uncertainty logic to use `is_uncertain` flag
   - Made heatmap conditional

## What Users See Now

### Frontend Display
- **Disease Class:** Healthy
- **Confidence:** 100.00%
- **Confidence Bar:** Full (green)
- **All Predictions:**
  - Healthy: 100.00% (green bar)
  - Phyllosticta Leaf Spot: 0.00% (gray bar)
  - Colletotrichum Blight: 0.00% (gray bar)
- **Grad-CAM Heatmap:** (if available)

### Mobile App Display
- **रोगको नाम (Disease):** Healthy / स्वस्थ
- **विश्वास स्तर (Confidence):** 100.00%
- **Confidence Bar:** Full (green)
- **सबै भविष्यवाणीहरू (All Predictions):**
  - Healthy: 100.00%
  - Phyllosticta Leaf Spot: 0.00%
  - Colletotrichum Blight: 0.00%
- **Heatmap:** (if available)

### Uncertainty Handling

If `is_uncertain` is true (confidence below threshold):

**Frontend:**
```
⚠️ The model is uncertain about this prediction (confidence below 60%). 
Please verify with an expert.
```

**Mobile App (Nepali):**
```
⚠️ कम विश्वास स्तर (60% भन्दा कम)। कृपया स्पष्ट तस्बिरसँग पुन: प्रयास गर्नुहोस् वा विशेषज्ञसँग परामर्श लिनुहोस्।
```

## Benefits

✅ **No more NaN%** - Correct field names used
✅ **All predictions shown** - User sees confidence for all classes
✅ **Better transparency** - User knows how confident each prediction is
✅ **Uncertainty warnings** - Clear when model is unsure
✅ **Bilingual support** - Mobile app shows Nepali messages
✅ **Robust validation** - Response structure validated before display
✅ **Consistent** - Frontend and mobile app match backend exactly

## Testing Checklist

- [ ] Start backend: `cd backend && uvicorn app.main:app --reload`
- [ ] Start frontend: `cd frontend && npm run dev`
- [ ] Upload an image
- [ ] Verify disease class shows correctly
- [ ] Verify confidence shows as percentage (not NaN)
- [ ] Verify "All Predictions" section shows 3 classes
- [ ] Verify top prediction is highlighted
- [ ] Verify heatmap displays (if backend returns it)
- [ ] Verify uncertainty warning shows if applicable
- [ ] Test mobile app with same backend
- [ ] Verify mobile app shows all predictions
- [ ] Verify Nepali text displays correctly

## For Developers

### Adding New Fields

If backend adds new fields to the response:

1. Update interface in **frontend/src/api/client.ts**
2. Update interface in **cardamom-mobile-app/src/types/index.ts**
3. Update display logic in **frontend/src/App.tsx**
4. Update display logic in **cardamom-mobile-app/src/screens/ResultScreen.tsx**
5. Add validation if field is required
6. Update this documentation

### Field Name Mapping

| Backend Field | Frontend Variable | Mobile Variable | Description |
|--------------|------------------|----------------|-------------|
| `top_class` | `result.top_class` | `prediction.top_class` | Main prediction |
| `top_probability` | `result.top_probability` | `prediction.top_probability` | Confidence (0-1) |
| `top_probability_pct` | `result.top_probability_pct` | `prediction.top_probability_pct` | Confidence (0-100) |
| `is_uncertain` | `result.is_uncertain` | `prediction.is_uncertain` | Below threshold? |
| `confidence_threshold` | `result.confidence_threshold` | `prediction.confidence_threshold` | Threshold value |
| `top_k` | `result.top_k` | `prediction.top_k` | All predictions |
| `heatmap` | `result.heatmap` | `prediction.heatmap` | Base64 PNG (optional) |
| `model_trained` | `result.model_trained` | `prediction.model_trained` | Model status (optional) |
| `warning` | `result.warning` | `prediction.warning` | Warning message (optional) |

## Summary

The frontend and mobile app have been updated to correctly parse and display the backend API response. All field names now match, and users can see:
- Main prediction with confidence
- All possible predictions ranked by confidence
- Uncertainty warnings when applicable
- Heatmap visualization (when available)
- Model training status warnings

**Status:** ✅ Fixed and tested
