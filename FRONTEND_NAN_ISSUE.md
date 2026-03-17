# Frontend NaN% Issue - Troubleshooting Guide

## Issue Description

When uploading an image in the frontend, you see:
- **Disease Class**: (empty or blank)
- **Confidence**: NaN%
- **Grad-CAM Heatmap**: Placeholder text or not showing

## Root Cause

The "NaN%" error occurs when the frontend cannot properly connect to the backend API or receives an invalid response.

### Most Common Causes (in order):

1. **Backend server is not running** (90% of cases)
2. **Wrong backend URL** 
3. **CORS issues**
4. **Backend error/crash**
5. **Network connectivity**

## Quick Fix (Most Likely Solution)

### Step 1: Start the Backend

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Start the server
uvicorn app.main:app --reload
```

You should see:
```
============================================================
Starting Cardamom Disease Detection API...
============================================================
✓ Using device: cpu
✓ Loading classifier model...
  → Classifier model loaded (untrained/placeholder)
============================================================
✓ API is ready to accept requests
============================================================
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Verify Backend is Running

Open in browser: **http://localhost:8000/health**

Expected response:
```json
{
  "status": "ok",
  "model_status": "untrained (placeholder)"
}
```

### Step 3: Try Frontend Again

1. Make sure frontend is running: `npm run dev` in `frontend/` directory
2. Open http://localhost:5173
3. Upload image
4. Click "Analyze"

## Detailed Troubleshooting

### Check #1: Is Backend Running?

**Test:**
```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{"status":"ok","model_status":"untrained (placeholder)"}
```

**If not working:**
- Backend is not running → Start it (see Step 1 above)
- Port 8000 is blocked → Check firewall
- Different port → Update frontend API URL in `frontend/src/api/client.ts`

### Check #2: Browser Console Logs

Open browser DevTools (F12) → Console tab

**Look for:**

**Success case:**
```
Sending image to API: image.jpg
Received response: {class_name: "...", confidence: 0.35, ...}
```

**Error cases:**

**Network Error:**
```
Prediction error: AxiosError: Network Error
Error details: {code: "ERR_NETWORK", ...}
```
→ **Fix**: Backend not running, start it

**CORS Error:**
```
Access to XMLHttpRequest at 'http://localhost:8000/predict' 
from origin 'http://localhost:5173' has been blocked by CORS policy
```
→ **Fix**: Backend should allow CORS for localhost:5173 (already configured)

**Timeout:**
```
Error details: {code: "ECONNABORTED", ...}
```
→ **Fix**: Backend took too long, might be using CPU (slow) or crashed

**Invalid Response:**
```
Prediction error: Error: Invalid response from server
```
→ **Fix**: Backend returned error, check backend console logs

### Check #3: Backend Console Logs

Look at the terminal where `uvicorn` is running.

**Normal:**
```
INFO:     127.0.0.1:52341 - "POST /predict HTTP/1.1" 200 OK
```

**Error:**
```
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  ...
```
→ **Fix**: Backend crashed, check the error message

### Check #4: Network Tab

Browser DevTools (F12) → Network tab → Filter by "XHR"

**Upload image and click Analyze**, you should see:

**Request:**
- Name: `predict`
- Status: `200` (success) or error code
- Method: `POST`
- Type: `xhr`

**Click on request → Preview tab:**

**Success:**
```json
{
  "class_name": "Colletotrichum Blight",
  "confidence": 0.3562,
  "heatmap": "iVBORw0KGg...",
  "model_trained": false,
  "warning": "⚠️ UNTRAINED MODEL: ..."
}
```

**Error:**
```json
{
  "detail": "Error processing image: ..."
}
```

## Common Error Messages

### "Cannot connect to server. Make sure the backend is running at http://localhost:8000"

**Cause**: Frontend can't reach backend

**Fix**:
1. Start backend: `cd backend && uvicorn app.main:app --reload`
2. Verify: `curl http://localhost:8000/health`
3. Check firewall settings

### "Request timeout. Please try again."

**Cause**: Backend took >30 seconds to respond

**Fix**:
1. Model inference might be slow on CPU
2. Check backend isn't frozen/crashed
3. Restart backend
4. Consider training on GPU if available

### "Invalid response from server: missing or invalid confidence value"

**Cause**: Backend returned unexpected data structure

**Fix**:
1. Check backend logs for errors
2. Make sure backend code is up to date
3. Backend might have crashed mid-request

### Error: "File must be an image"

**Cause**: Uploaded file is not an image

**Fix**:
1. Make sure file is .jpg, .png, etc.
2. Don't upload .txt, .pdf, or other non-image files

## Understanding the Response

### Untrained Model (Expected)

If you just set up the project, the model is **untrained** (random weights):

```json
{
  "class_name": "Colletotrichum Blight",
  "confidence": 0.3562,  // ~33% (random guess for 3 classes)
  "model_trained": false,
  "warning": "⚠️ UNTRAINED MODEL: This prediction uses a placeholder model..."
}
```

**This is NORMAL**. The predictions are random (~33% confidence) because the model needs training.

→ See `TRAIN_NOW.md` or `WHY_LOW_ACCURACY.md` for training instructions.

### Trained Model (After Training)

After training the model:

```json
{
  "class_name": "Phyllosticta Leaf Spot",
  "confidence": 0.9234,  // ~92% (high confidence)
  "model_trained": true,
  "warning": null
}
```

## Step-by-Step Debugging Checklist

- [ ] Backend running? (`uvicorn app.main:app --reload`)
- [ ] Backend accessible? (http://localhost:8000/health shows "ok")
- [ ] Frontend running? (`npm run dev` in frontend/)
- [ ] Browser console shows request being sent?
- [ ] Browser console shows "Received response"?
- [ ] Network tab shows request to `/predict`?
- [ ] Network tab shows status 200?
- [ ] Response contains `class_name`, `confidence`, `heatmap`?
- [ ] `confidence` is a number (not null/undefined)?

## Quick Test Commands

### Test Backend Health
```bash
curl http://localhost:8000/health
```

### Test Prediction (with test image)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

### Start Backend (Full Command)
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend
```bash
cd frontend
npm run dev
```

## Still Not Working?

### Collect Debug Information:

1. **Backend logs** (from terminal running uvicorn)
2. **Browser console logs** (F12 → Console, copy all errors)
3. **Network request details** (F12 → Network → /predict → Preview)
4. **Your setup**:
   - OS (Windows/Mac/Linux)
   - Python version: `python --version`
   - Node version: `node --version`
   - Backend running? Yes/No
   - Frontend running? Yes/No

### Then:
- Check the error message carefully
- Search for similar error in GitHub issues
- Create new issue with debug information above

## Summary

**The NaN% issue happens when frontend can't get valid data from backend.**

**Most common fix**: Start the backend server!

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

Then try uploading image again. It should work! ✅

## Related Documentation

- `START_HERE.md` - First-time setup
- `AFTER_PULL.md` - After pulling code changes
- `INSTALL_DEPENDENCIES.md` - Installing required packages
- `TRAIN_NOW.md` - Training the model
- `WHY_LOW_ACCURACY.md` - Understanding low confidence (~33%)
