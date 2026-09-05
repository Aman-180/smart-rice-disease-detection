import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# -----------------------------
# Basic paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "classification")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_rice.pth")


# -----------------------------
# Training settings
# -----------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 2 #10
LEARNING_RATE = 0.001
NUM_CLASSES = 5


def get_device():
    """Use GPU if available, otherwise use CPU."""
    if torch.cuda.is_available():
        print("GPU detected. Training on CUDA.")
        return torch.device("cuda")

    print("GPU not detected. Training on CPU.")
    return torch.device("cpu")


def create_data_loaders():
    """Create train and validation data loaders."""

    # Training images get augmentation to make model more robust.
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation images should not be heavily augmented.
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Classes found:")
    print(train_dataset.classes)
    print("Class to index mapping:")
    print(train_dataset.class_to_idx)

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset.classes


def create_model(device):
    """Load pretrained ResNet-50 and replace the final layer."""

    print("Loading pretrained ResNet-50...")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # ResNet-50 originally predicts 1000 ImageNet classes.
    # We replace the last layer so it predicts 5 rice classes.
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, NUM_CLASSES)

    model = model.to(device)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and update weights.
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    """Check model performance on validation data."""

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_accuracy = correct / total

    return val_loss, val_accuracy


def main():
    print("Starting ResNet-50 rice disease training...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()
    train_loader, val_loader, class_names = create_data_loaders()

    model = create_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy * 100:.2f}%")

        # Save only the best model.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": train_loader.dataset.class_to_idx,
                "val_accuracy": best_val_accuracy
            }

            torch.save(checkpoint, MODEL_PATH)
            print(f"Best model saved to: {MODEL_PATH}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()