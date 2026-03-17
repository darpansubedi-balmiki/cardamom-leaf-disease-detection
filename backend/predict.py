"""
Predict cardamom leaf disease class for images in a folder.

Usage:
  python predict.py --input external_test
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class Config:
    MODEL_PATH = "models/cardamom_model.pt"
    IMG_SIZE = 224
    NUM_CLASSES = 3
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # IMPORTANT: must match training class order used by ImageFolder.
    # We'll print this order so you can verify it matches your dataset/classes.
    CLASS_NAMES = ["colletotrichum_blight", "healthy", "phyllosticta_leaf_spot"]


def create_model():
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, Config.NUM_CLASSES),
    )
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def is_image(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing images")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    model_path = Path(Config.MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path} (train first: python train.py)")

    print(f"Device: {Config.DEVICE}")
    print(f"Model: {model_path}")
    print(f"Class order used for predictions: {Config.CLASS_NAMES}")

    model = create_model().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    tfm = get_transform()

    images = sorted([p for p in input_dir.rglob("*") if p.is_file() and is_image(p)])
    if not images:
        raise SystemExit(f"No images found in: {input_dir}")

    with torch.no_grad():
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(Config.DEVICE)

            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            pred_name = Config.CLASS_NAMES[pred_idx]
            conf = float(probs[pred_idx])

            print(f"{img_path} -> {pred_name} (confidence={conf*100:.2f}%)")


if __name__ == "__main__":
    main()