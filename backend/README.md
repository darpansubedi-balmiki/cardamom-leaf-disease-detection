# Backend - Cardamom Leaf Disease Detection

FastAPI backend for cardamom leaf disease classification with PyTorch.

## Features

- **Disease Classification**: Classifies images into 3 categories:
  - Colletotrichum Blight
  - Phyllosticta Leaf Spot
  - Healthy

- **Grad-CAM Visualization**: Generates heatmaps showing which regions of the leaf influenced the prediction

- **Background Removal**: Placeholder for U2-Net-based background removal (to be integrated with trained model)

- **CORS Enabled**: Configured for frontend communication

## Project Structure

```
backend/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── schemas.py                 # Pydantic models for request/response
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cardamom_model.py      # CNN classifier (placeholder)
│   │   └── u2net_segmenter.py     # U2-Net background removal (placeholder)
│   └── utils/
│       ├── __init__.py
│       ├── image_preprocess.py    # Image preprocessing utilities
│       ├── grad_cam.py            # Grad-CAM implementation
│       └── overlay.py             # Heatmap overlay utilities
├── models/                        # Directory for trained model weights
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

Start the development server:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /predict

Upload an image and get disease prediction with Grad-CAM visualization.

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

## Model Integration

### Current State (Placeholder)

The current implementation uses randomly initialized models for demonstration:
- **Classifier**: Simple 4-layer CNN with random weights
- **U2-Net**: Placeholder that passes images through unchanged

### Integrating Trained Models

When you have trained models:

1. **Place trained weights** in the `backend/models/` directory:
   - `cardamom_model.pt` - Trained classifier (e.g., EfficientNetV2)
   - `u2net.pth` - Trained U2-Net for background removal

2. **Update model loading** in `app/models/`:
   - Modify `cardamom_model.py` to load your actual architecture
   - Implement U2-Net loading in `u2net_segmenter.py`

3. **Adjust preprocessing** if needed in `app/utils/image_preprocess.py`

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest
```

### Code Style

The project follows PEP 8 style guidelines. Format code with:
```bash
pip install black
black app/
```

## Troubleshooting

### CUDA/GPU Issues

If you encounter GPU-related errors, the model will automatically fall back to CPU. To force CPU mode:
```python
device = 'cpu'
```

### Module Import Errors

Make sure you're running the server from the `backend` directory, or add it to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/backend"
```

## License

MIT
