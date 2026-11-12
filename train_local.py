
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from xray.ml.model.arch import Net


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = Path(r"C:\Users\nikhi\Downloads\Data")

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_PATH = PROJECT_DIR / "xray_model.pth"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==========================================
# 3. IMAGE TRANSFORMATIONS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=7),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. CUSTOM DATASET
# ==========================================

class XRayDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================
# 5. COLLECT TWO CLASSES ONLY
# ==========================================

def collect_samples(folder):

    samples = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {class_folder}"
            )

        image_files = sorted([
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in
            [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        if not image_files:
            raise ValueError(
                f"No images found in {class_folder}"
            )

        for image_path in image_files:
            samples.append((image_path, label))

        print(
            f"{folder.name} - {class_name}: "
            f"{len(image_files)} images"
        )

    return samples


# ==========================================
# 6. TRAIN / VALIDATION SPLIT
# ==========================================

def split_training_samples(samples):

    train_samples = []
    validation_samples = []

    for label in range(len(CLASS_NAMES)):

        class_samples = [
            sample for sample in samples
            if sample[1] == label
        ]

        random.shuffle(class_samples)

        validation_count = max(
            1,
            int(len(class_samples) * VALIDATION_RATIO)
        )

        validation_samples.extend(
            class_samples[:validation_count]
        )

        train_samples.extend(
            class_samples[validation_count:]
        )

    random.shuffle(train_samples)
    random.shuffle(validation_samples)

    return train_samples, validation_samples


# ==========================================
# 7. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


# ==========================================
# 8. VALIDATION FUNCTION
# ==========================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


# ==========================================
# 9. FINAL TEST EVALUATION
# ==========================================

def test_model(model, loader):

    model.eval()

    matrix = [[0, 0], [0, 0]]

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            for actual, predicted in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist()
            ):
                matrix[actual][predicted] += 1

    total = sum(sum(row) for row in matrix)

    correct = matrix[0][0] + matrix[1][1]

    print("\n========== FINAL TEST RESULTS ==========")

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    print("\nConfusion Matrix")
    print("Rows = Actual | Columns = Predicted")
    print("                 NORMAL  PNEUMONIA")

    for index, row in enumerate(matrix):
        print(f"{CLASS_NAMES[index]:15} {row}")

    for index, class_name in enumerate(CLASS_NAMES):

        actual_count = sum(matrix[index])

        recall = (
            100 * matrix[index][index] / actual_count
            if actual_count else 0
        )

        print(f"{class_name} Recall: {recall:.2f}%")


# ==========================================
# 10. MAIN TRAINING PIPELINE
# ==========================================

def main():

    print("\n====================================")
    print("LUNG DISEASE CLASSIFICATION")
    print("NORMAL vs PNEUMONIA")
    print("====================================")

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(TRAIN_DIR)

    if not TEST_DIR.exists():
        raise FileNotFoundError(TEST_DIR)

    # Collect only NORMAL and PNEUMONIA.
    all_train_samples = collect_samples(TRAIN_DIR)
    test_samples = collect_samples(TEST_DIR)

    train_samples, validation_samples = (
        split_training_samples(all_train_samples)
    )

    print("\nDATASET SUMMARY")
    print(f"Training images: {len(train_samples)}")
    print(f"Validation images: {len(validation_samples)}")
    print(f"Testing images: {len(test_samples)}")

    train_dataset = XRayDataset(
        train_samples,
        train_transform
    )

    validation_dataset = XRayDataset(
        validation_samples,
        test_transform
    )

    test_dataset = XRayDataset(
        test_samples,
        test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Class imbalance handling.
    class_counts = [
        sum(1 for _, label in train_samples if label == index)
        for index in range(2)
    ]

    total_samples = sum(class_counts)

    class_weights = torch.tensor(
        [
            total_samples / (2 * count)
            for count in class_counts
        ],
        dtype=torch.float32
    ).to(DEVICE)

    print("\nCLASS WEIGHTS")

    for name, weight in zip(CLASS_NAMES, class_weights):
        print(f"{name}: {weight.item():.4f}")

    # Initialize model.
    model = Net().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_validation_loss = float("inf")
    best_epoch = 0

    patience = 4
    no_improvement = 0

    # Training loop.
    for epoch in range(EPOCHS):

        print(
            f"\n========== EPOCH {epoch + 1}/{EPOCHS} =========="
        )

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = evaluate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(val_loss)

        print(
            f"Training Loss: {train_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Save the best model.
        if val_loss < best_validation_loss:

            best_validation_loss = val_loss
            best_epoch = epoch + 1
            no_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("Best model saved successfully.")

        else:
            no_improvement += 1

        if no_improvement >= patience:
            print("\nEarly stopping activated.")
            break

    # Load the best checkpoint.
    print(f"\nBest epoch: {best_epoch}")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    # Final evaluation on separate test data.
    test_model(model, test_loader)

    print("\n====================================")
    print("TRAINING COMPLETED")
    print(f"Model saved at: {MODEL_PATH}")
    print("NORMAL = 0")
    print("PNEUMONIA = 1")
    print("COVID19 = EXCLUDED")
    print("====================================")


if __name__ == "__main__":
    main()
