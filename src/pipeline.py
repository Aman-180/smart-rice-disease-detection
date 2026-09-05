import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Reuse existing classification code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import (
    get_device,
    load_model,
    predict,
    print_result,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YOLO_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "runs", "detect", "runs", "detect",
    "rice_disease_yolo_continued", "weights", "best.pt"
)

CONF_THRESHOLD = 0.25


def prepare_cropped_image(pil_image):
    """Convert an in-memory cropped PIL image into ResNet input format."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def detect_disease_region(yolo_model, image_path):
    """Run YOLO on the image, return the best-confidence cropped region as a PIL image.
    If no detection found, return the original full image as fallback."""

    results = yolo_model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]
    original_image = Image.open(image_path).convert("RGB")

    if len(result.boxes) == 0:
        print("No diseased region detected by YOLO. Using full image as fallback.")
        return original_image, None

    # pick the highest-confidence box
    confidences = result.boxes.conf.tolist()
    best_idx = confidences.index(max(confidences))
    box = result.boxes.xyxy[best_idx].tolist()  # [x1, y1, x2, y2]
    box_conf = confidences[best_idx]

    x1, y1, x2, y2 = [int(coord) for coord in box]
    cropped = original_image.crop((x1, y1, x2, y2))

    print(f"Diseased region detected (YOLO confidence: {box_conf * 100:.2f}%)")
    return cropped, box


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: YOLO detects diseased region, ResNet classifies it."
    )
    parser.add_argument("--image", required=True, help="Path to the rice leaf image")
    args = parser.parse_args()

    print("Loading models...")
    device = get_device()
    classifier_model, class_names = load_model(device)
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print(f"\nProcessing image: {args.image}")
    print("Step 1: Detecting diseased region with YOLO...")
    cropped_image, box = detect_disease_region(yolo_model, args.image)

    print("Step 2: Classifying region with ResNet-50...")
    image_tensor = prepare_cropped_image(cropped_image)

    results = predict(
        model=classifier_model,
        image_tensor=image_tensor,
        class_names=class_names,
        device=device
    )

    print_result(results)


if __name__ == "__main__":
    main()