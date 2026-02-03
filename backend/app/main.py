"""FastAPI application for cardamom leaf disease detection."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from io import BytesIO
import traceback

from app.schemas import HealthResponse, PredictionResponse
from app.models.cardamom_model import load_model
from app.models.u2net_segmenter import load_u2net, apply_background_removal
from app.utils.image_preprocess import preprocess_image
from app.utils.grad_cam import generate_gradcam_heatmap
from app.utils.overlay import overlay_and_encode


# Initialize FastAPI app
app = FastAPI(
    title="Cardamom Leaf Disease Detection API",
    description="API for detecting diseases in cardamom leaves using deep learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # Alternative React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names
CLASS_NAMES = ["Colletotrichum Blight", "Phyllosticta Leaf Spot", "Healthy"]

# Global variables for models
classifier_model = None
u2net_model = None
device = None
model_is_trained = False  # Track if using trained or placeholder model


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global classifier_model, u2net_model, device, model_is_trained
    
    print("Starting up application...")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load classifier model
    print("Loading classifier model...")
    classifier_model, model_is_trained = load_model(device=device)
    print("Classifier model loaded successfully")
    
    # Load U2-Net model (placeholder for now)
    print("Loading U2-Net model...")
    u2net_model = load_u2net(device=device)
    print("U2-Net model loaded (placeholder)")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns:
        Status message indicating the API is running
    """
    model_status = "trained" if model_is_trained else "untrained (placeholder)"
    return HealthResponse(status="ok", model_status=model_status)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict disease class for an uploaded cardamom leaf image.
    
    Args:
        file: Uploaded image file (multipart/form-data)
        
    Returns:
        Prediction result with class name, confidence, and Grad-CAM heatmap
        
    Raises:
        HTTPException: If image processing or prediction fails
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Apply background removal (placeholder - just passes through)
        image = apply_background_removal(image, u2net_model)
        
        # Preprocess image for model
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = classifier_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get class name
        class_name = CLASS_NAMES[predicted_class]
        
        # Generate Grad-CAM heatmap
        # Target the last convolutional layer (conv4)
        # Get the last convolutional layer from EfficientNetV2
        # EfficientNetV2 structure: backbone.features[-1] is the last conv block
        target_layer = None
        for module in classifier_model.backbone.features[-1].modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            # Fallback: find any last Conv2d layer
            for name, module in classifier_model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    
        heatmap = generate_gradcam_heatmap(
            classifier_model,
            input_tensor,
            target_layer,
            target_class=predicted_class
        )
        
        # Overlay heatmap on original image and encode as base64
        heatmap_base64 = overlay_and_encode(image, heatmap, alpha=0.4)
        
        # Add warning if using untrained model
        warning = None
        if not model_is_trained:
            warning = (
                "⚠️ UNTRAINED MODEL: This prediction uses a placeholder model with random weights. "
                "Predictions are not accurate. Please train the model with real data for production use. "
                "See MODEL_TRAINING.md for instructions."
            )
        
        return PredictionResponse(
            class_name=class_name,
            confidence=confidence,
            heatmap=heatmap_base64,
            model_trained=model_is_trained,
            warning=warning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
