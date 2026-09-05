import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# -----------------------------
# Basic paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classification",
    "resnet50_rice.pth"
)

BATCH_SIZE = 16
NUM_CLASSES = 5


def get_device():
    """Use GPU if available, otherwise use CPU."""
    if torch.cuda.is_available():
        print("GPU detected. Evaluating on CUDA.")
        return torch.device("cuda")

    print("GPU not detected. Evaluating on CPU.")
    return torch.device("cpu")


def create_test_loader():
    """Load test images with the same preprocessing used during validation."""

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Test classes found:")
    print(test_dataset.classes)

    print("Class to index mapping:")
    print(test_dataset.class_to_idx)

    print(f"Test images: {len(test_dataset)}")

    return test_loader, test_dataset.classes


def load_model(device):
    """Load the trained ResNet-50 model from disk."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    print(f"Loading model from: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = models.resnet50(weights=None)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, NUM_CLASSES)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    model.eval()

    return model


def evaluate(model, test_loader, class_names, device):
    """Evaluate model accuracy on the test set."""

    total_correct = 0
    total_images = 0

    class_correct = [0 for _ in class_names]
    class_total = [0 for _ in class_names]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

            for i in range(labels.size(0)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()

                class_total[true_label] += 1

                if predicted_label == true_label:
                    class_correct[true_label] += 1

    overall_accuracy = total_correct / total_images

    print("\nEvaluation complete.")
    print(f"Overall Test Accuracy: {overall_accuracy * 100:.2f}%")

    print("\nClass-wise Accuracy:")

    for i, class_name in enumerate(class_names):
        if class_total[i] == 0:
            accuracy = 0
        else:
            accuracy = class_correct[i] / class_total[i]

        print(
            f"{class_name}: "
            f"{accuracy * 100:.2f}% "
            f"({class_correct[i]}/{class_total[i]})"
        )


def main():
    print("Starting ResNet-50 evaluation...")

    device = get_device()
    test_loader, class_names = create_test_loader()

    model = load_model(device)

    evaluate(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device
    )


if __name__ == "__main__":
    main()