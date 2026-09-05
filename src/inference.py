import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# -----------------------------
# Basic paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classification",
    "resnet50_rice.pth"
)

NUM_CLASSES = 5


# Simple treatment advice for demo
TREATMENTS = {
    "bacterialblight": [
        "Avoid excess nitrogen fertilizer.",
        "Use clean water and improve field drainage.",
        "Remove heavily infected plant material.",
        "Consult a local agriculture officer before spraying chemicals."
    ],
    "blast": [
        "Avoid excess nitrogen fertilizer.",
        "Keep proper spacing between plants.",
        "Remove infected crop residue after harvest.",
        "Use recommended fungicide only after expert advice."
    ],
    "brownspot": [
        "Improve soil nutrition, especially potassium and silicon.",
        "Use disease-free seeds.",
        "Avoid water stress in the field.",
        "Consult a local agriculture officer for suitable fungicide advice."
    ],
    "healthy": [
        "No disease detected.",
        "Continue regular monitoring.",
        "Maintain balanced fertilizer and proper irrigation."
    ],
    "tungro": [
        "Control green leafhopper insects.",
        "Remove infected plants early.",
        "Avoid planting near infected fields.",
        "Use resistant rice varieties if available."
    ]
}


def get_device():
    """Use GPU if available, otherwise use CPU."""
    if torch.cuda.is_available():
        print("GPU detected. Running inference on CUDA.")
        return torch.device("cuda")

    print("GPU not detected. Running inference on CPU.")
    return torch.device("cpu")


def load_model(device):
    """Load the trained ResNet-50 model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]

    model = models.resnet50(weights=None)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, NUM_CLASSES)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    model.eval()

    return model, class_names


def prepare_image(image_path):
    """Open image and convert it into the format ResNet expects."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image)

    # Model expects a batch of images.
    # This changes shape from [3, 224, 224] to [1, 3, 224, 224].
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict(model, image_tensor, class_names, device):
    """Predict class and confidence for one image."""

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=3)

    results = []

    for i in range(3):
        class_index = top_indices[0][i].item()
        confidence = top_probs[0][i].item()
        class_name = class_names[class_index]

        results.append((class_name, confidence))

    return results


def print_result(results):
    """Print prediction results and treatment advice."""

    top_class, top_confidence = results[0]

    print("\nPrediction Result")
    print("-----------------")
    print(f"Disease/Class: {top_class}")
    print(f"Confidence: {top_confidence * 100:.2f}%")

    print("\nTop 3 Predictions:")
    for rank, (class_name, confidence) in enumerate(results, start=1):
        print(f"{rank}. {class_name}: {confidence * 100:.2f}%")

    if top_confidence < 0.60:
        print("\nWarning:")
        print("Confidence is low. Please upload a clearer, closer photo.")

    print("\nSuggested Advice:")
    for advice in TREATMENTS.get(top_class, []):
        print(f"- {advice}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict rice disease from one image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the rice leaf image"
    )

    args = parser.parse_args()

    print("Starting single-image inference...")

    device = get_device()
    model, class_names = load_model(device)

    image_tensor = prepare_image(args.image)

    results = predict(
        model=model,
        image_tensor=image_tensor,
        class_names=class_names,
        device=device
    )

    print_result(results)


if __name__ == "__main__":
    main()