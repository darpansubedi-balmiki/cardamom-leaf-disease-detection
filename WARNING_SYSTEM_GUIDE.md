# Visual Guide: New Warning System

## What You'll See Now

### 1. Backend Startup (Console Output)

**When starting the backend server:**

```bash
$ uvicorn app.main:app --reload

INFO:     Will watch for changes in these directories: ['/path/to/backend']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
Starting up application...
Using device: cpu
Loading classifier model...
⚠️  Model file not found at models/cardamom_model.pt
⚠️  Using randomly initialized weights (UNTRAINED PLACEHOLDER MODEL)
⚠️  Predictions will be inaccurate - please train the model first!
⚠️  See MODEL_TRAINING.md for training instructions
Classifier model loaded successfully
Loading U2-Net model...
U2-Net model loaded (placeholder)
INFO:     Application startup complete.
```

### 2. API Response (JSON Output)

**When making a prediction with untrained model:**

```bash
$ curl -X POST -F "file=@leaf_image.jpg" http://localhost:8000/predict
```

**Response:**
```json
{
  "class_name": "Colletotrichum Blight",
  "confidence": 0.3562,
  "heatmap": "iVBORw0KGgoAAAANSUhEUgAA...(base64 string)...",
  "model_trained": false,
  "warning": "⚠️ UNTRAINED MODEL: This prediction uses a placeholder model with random weights. Predictions are not accurate. Please train the model with real data for production use. See MODEL_TRAINING.md for instructions."
}
```

### 3. Health Check Response

```bash
$ curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_status": "untrained (placeholder)"
}
```

## After Training the Model

### 1. Backend Startup (With Trained Model)

```bash
$ uvicorn app.main:app --reload

Starting up application...
Using device: cpu
Loading classifier model...
✓ Loaded trained model weights from models/cardamom_model.pt
Classifier model loaded successfully
Loading U2-Net model...
U2-Net model loaded (placeholder)
INFO:     Application startup complete.
```

### 2. API Response (With Trained Model)

```json
{
  "class_name": "Phyllosticta Leaf Spot",
  "confidence": 0.9523,
  "heatmap": "iVBORw0KGgoAAAANSUhEUgAA...(base64 string)...",
  "model_trained": true,
  "warning": null
}
```

Notice:
- ✅ Correct disease classification
- ✅ High confidence (95.23%)
- ✅ `model_trained: true`
- ✅ No warning message

### 3. Health Check (With Trained Model)

```json
{
  "status": "ok",
  "model_status": "trained"
}
```

## Mobile App Integration

The mobile app and web frontend can now check the `model_trained` field:

```typescript
// In your mobile app or frontend
const result = await api.predictDisease(imageFile);

if (!result.model_trained) {
  // Show warning to user
  Alert.alert(
    "Untrained Model",
    result.warning,
    [{ text: "OK" }]
  );
}
```

## Comparison Table

| Metric | Untrained Model (Current) | Trained Model (After Training) |
|--------|---------------------------|--------------------------------|
| **Confidence** | 20-40% | 80-95%+ |
| **Accuracy** | ~33% (random) | 90%+ |
| **Correct Predictions** | Unlikely | High probability |
| **API Field** | `model_trained: false` | `model_trained: true` |
| **Warning** | Shows warning message | `warning: null` |
| **Model Status** | "untrained (placeholder)" | "trained" |
| **Console Output** | ⚠️ warnings | ✓ success messages |

## Frontend Display Suggestions

### Show Warning Banner

```jsx
{!result.model_trained && (
  <div className="warning-banner">
    <strong>⚠️ Notice:</strong> Using untrained model. 
    Predictions are not accurate. 
    <a href="/docs/training">Learn about training</a>
  </div>
)}
```

### Confidence Color Coding

```jsx
const getConfidenceColor = (confidence, isTrained) => {
  if (!isTrained) return 'red';  // Always red for untrained
  if (confidence > 0.8) return 'green';
  if (confidence > 0.6) return 'orange';
  return 'red';
};
```

### Mobile App Alert

```typescript
// React Native Mobile App
if (!prediction.model_trained) {
  Alert.alert(
    'मोडल प्रशिक्षित छैन',  // Nepali: Model Not Trained
    'यो भविष्यवाणी अप्रशिक्षित मोडल प्रयोग गरेर गरिएको छ। परिणाम सही नहुन सक्छ।',
    // This prediction uses an untrained model. Results may not be accurate.
    [
      { text: 'बुझें', style: 'default' }  // Understood
    ]
  );
}
```

## API Documentation

The OpenAPI/Swagger docs at `http://localhost:8000/docs` will now show:

**PredictionResponse Schema:**
```json
{
  "class_name": "string",
  "confidence": "number (0-1)",
  "heatmap": "string (base64)",
  "model_trained": "boolean",
  "warning": "string | null"
}
```

**HealthResponse Schema:**
```json
{
  "status": "string",
  "model_status": "string | null"
}
```

## Summary

The new warning system makes it impossible to miss that the model needs training:

1. ⚠️ Console warnings when starting server
2. ⚠️ API response includes `model_trained: false`
3. ⚠️ Explicit warning message in response
4. ⚠️ Health endpoint shows model status
5. ⚠️ Documentation clearly explains training need

**Next Step**: Follow [MODEL_TRAINING.md](MODEL_TRAINING.md) to train your model with real data!
