"""
ResNet-50 training script for the Tomato dataset.

Default task:
    Predict Side Case: clear vs defected

The Excel annotation file contains image names such as:
    T0001_E, T0001_S1, T0001_S2, T0001_T

The script:
1. Reads annotations from the .xls file.
2. Finds matching images recursively in:
       Manualy Segmented/
       Unsegmented/
3. Removes rows without a target label.
4. Splits by TOMATO ID (T0001, T0002, ...) so different views of the
   same tomato cannot leak into train/validation/test.
5. Fine-tunes pretrained ResNet-50.
6. Saves the best model and test metrics.

Run from VS Code terminal:
    python resnet50_tomato.py

You can change TARGET below to:
    "side_case"   -> clear / defected
    "ripeness"    -> ripe / unripe

Install first:
    pip install torch torchvision pandas xlrd scikit-learn pillow matplotlib
"""

from pathlib import Path
import copy
import random
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


# ============================================================
# 1. CONFIGURATION - CHANGE THESE PATHS
# ============================================================

# Example Windows structure:
#
# tomato_dataset/
# ├── Manualy Segmented/
# │   ├── T0001_E.jpg
# │   ├── T0001_S1.jpg
# │   └── ...
# ├── Unsegmented/
# │   ├── T0001_E.jpg
# │   └── ...
# └── Dataset_anotations.xls

DATASET_ROOT = Path(r"C:\Users\arafa\Documents\Thesis\Tomato body Dataset")
ANNOTATION_FILE = DATASET_ROOT / "Dataset_anotations.xls"

# Choose the prediction target:
# "side_case" -> clear / defected
# "ripeness"  -> ripe / unripe
TARGET = "side_case"

IMAGE_FOLDERS = [
    DATASET_ROOT / "Manualy Segmented",
    DATASET_ROOT / "Unsegmented",
]

OUTPUT_DIR = DATASET_ROOT / "resnet50_output"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# Set True if you have an NVIDIA GPU.
# The script will automatically use CUDA if available.
USE_AMP = True


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("Tomato ResNet-50 training")
print("=" * 70)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Target: {TARGET}")


# ============================================================
# 3. READ EXCEL ANNOTATIONS
# ============================================================

if not ANNOTATION_FILE.exists():
    raise FileNotFoundError(
        f"Annotation file not found:\n{ANNOTATION_FILE}\n\n"
        "Change DATASET_ROOT at the top of this file."
    )

print("\nReading Excel annotations...")

# Your workbook has the actual column headers on row 4.
# header=3 means pandas uses Excel row 4 as the header.
df = pd.read_excel(ANNOTATION_FILE, sheet_name="Annotations", header=3)

df.columns = [str(c).strip() for c in df.columns]

required_columns = ["Name", "Ripeness", "Side Case"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' was not found in the Excel file.\n"
            f"Found columns: {df.columns.tolist()}"
        )


# ============================================================
# 4. SELECT TARGET LABEL
# ============================================================

if TARGET == "side_case":
    target_column = "Side Case"

    # Normalize spelling/capitalization.
    df["label"] = (
        df[target_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan})
    )

    # Keep only labeled examples.
    df = df[df["label"].isin(["clear", "defected"])].copy()

    class_names = ["clear", "defected"]

elif TARGET == "ripeness":
    target_column = "Ripeness"

    df["label"] = (
        df[target_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan})
    )

    df = df[df["label"].isin(["ripe", "unripe"])].copy()

    class_names = ["ripe", "unripe"]

else:
    raise ValueError("TARGET must be 'side_case' or 'ripeness'.")


df["image_name"] = df["Name"].astype(str).str.strip()

# Tomato group:
# T0001_E -> T0001
# T0001_S1 -> T0001
# T0001_S2 -> T0001
# T0001_T -> T0001
df["tomato_id"] = df["image_name"].str.extract(r"^(T\d+)", expand=False)

df = df.dropna(subset=["tomato_id"])

print(f"Annotated rows for target '{TARGET}': {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 5. FIND ALL IMAGES
# ============================================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\nSearching for images...")

image_map = {}

for folder in IMAGE_FOLDERS:
    if not folder.exists():
        print(f"WARNING: folder does not exist: {folder}")
        continue

    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
            # Example: T0001_E.jpg -> T0001_E
            stem = file.stem.strip()
            image_map[stem] = file

print(f"Images found: {len(image_map)}")

if len(image_map) == 0:
    raise RuntimeError(
        "No images found. Check IMAGE_FOLDERS in the configuration."
    )

df["image_path"] = df["image_name"].map(image_map)

missing = df["image_path"].isna().sum()

if missing > 0:
    print(f"WARNING: {missing} annotated images were not found.")
    print("Examples of missing images:")
    print(df.loc[df["image_path"].isna(), "image_name"].head(20).tolist())

df = df.dropna(subset=["image_path"]).copy()

print(f"Usable labeled images: {len(df)}")


# ============================================================
# 6. GROUPED TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# IMPORTANT:
# A tomato has multiple views:
#   E, S1, S2, T
#
# We split by tomato_id instead of individual images.
# Therefore, all four views of T0001 stay in the same split.
# This prevents data leakage.

groups = df["tomato_id"].unique()

train_groups, temp_groups = train_test_split(
    groups,
    test_size=(1 - TRAIN_RATIO),
    random_state=SEED,
)

relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=relative_test_size,
    random_state=SEED,
)

train_df = df[df["tomato_id"].isin(train_groups)].copy()
val_df = df[df["tomato_id"].isin(val_groups)].copy()
test_df = df[df["tomato_id"].isin(test_groups)].copy()

print("\nSplit:")
print(f"Train tomatoes: {len(train_groups)}, images: {len(train_df)}")
print(f"Val   tomatoes: {len(val_groups)}, images: {len(val_df)}")
print(f"Test  tomatoes: {len(test_groups)}, images: {len(test_df)}")

print("\nTrain class distribution:")
print(train_df["label"].value_counts())

print("\nValidation class distribution:")
print(val_df["label"].value_counts())

print("\nTest class distribution:")
print(test_df["label"].value_counts())


# ============================================================
# 7. PYTORCH DATASET
# ============================================================

label_to_index = {
    name: i for i, name in enumerate(class_names)
}


class TomatoDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = Path(row["image_path"])
        label = label_to_index[row["label"]]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Could not open image: {image_path}\n{e}"
            )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ImageNet normalization because we use pretrained ResNet-50.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


train_dataset = TomatoDataset(train_df, train_transform)
val_dataset = TomatoDataset(val_df, eval_transform)
test_dataset = TomatoDataset(test_df, eval_transform)


num_workers = 0  # Safe for Windows/VS Code.

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(DEVICE.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# 8. CREATE PRETRAINED RESNET-50
# ============================================================

print("\nLoading pretrained ResNet-50...")

try:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
except AttributeError:
    # Compatibility with older torchvision versions.
    model = models.resnet50(pretrained=True)

# Replace the final classification layer.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model = model.to(DEVICE)

print(f"Classes: {class_names}")
print(f"Final layer: {num_features} -> {len(class_names)}")


# ============================================================
# 9. CLASS-WEIGHTED LOSS
# ============================================================
# Helps if clear/defected are not perfectly balanced.

counts = train_df["label"].value_counts()

class_weights = []
for class_name in class_names:
    class_weights.append(len(train_df) / (len(class_names) * counts[class_name]))

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# 10. TRAIN / VALIDATION FUNCTIONS
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(USE_AMP and DEVICE.type == "cuda"),
)


def train_one_epoch():
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(USE_AMP and DEVICE.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = total_loss / len(train_dataset)
    epoch_acc = accuracy_score(all_labels, all_predictions)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(loader, dataset):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(dataset)
    acc = accuracy_score(all_labels, all_predictions)

    return loss, acc, all_labels, all_predictions


# ============================================================
# 11. TRAINING
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
best_model_state = None
patience = 5
epochs_without_improvement = 0

history = []

print("\nStarting training...")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc, _, _ = evaluate(
        val_loader,
        val_dataset,
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": best_model_state,
                "class_names": class_names,
                "target": TARGET,
                "image_size": IMAGE_SIZE,
                "model": "resnet50",
            },
            OUTPUT_DIR / "best_resnet50.pth",
        )

        print("  -> Best model saved.")

    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break


# ============================================================
# 12. TEST THE BEST MODEL
# ============================================================

print("\nLoading best model...")
model.load_state_dict(best_model_state)

test_loss, test_acc, y_true, y_pred = evaluate(
    test_loader,
    test_dataset,
)

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

print("\nClassification report:")
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
)
print(report)

print("Confusion matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

history_df = pd.DataFrame(history)
history_df.to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False,
)

with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Target: {TARGET}\n")
    f.write(f"Classes: {class_names}\n")
    f.write(f"Test loss: {test_loss:.6f}\n")
    f.write(f"Test accuracy: {test_acc:.6f}\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(np.array2string(cm))

with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "dataset_root": str(DATASET_ROOT),
            "annotation_file": str(ANNOTATION_FILE),
            "target": TARGET,
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        },
        f,
        indent=4,
    )

print("\nSaved files:")
print(f"  {OUTPUT_DIR / 'best_resnet50.pth'}")
print(f"  {OUTPUT_DIR / 'training_history.csv'}")
print(f"  {OUTPUT_DIR / 'classification_report.txt'}")
print(f"  {OUTPUT_DIR / 'config.json'}")

print("\nTraining finished.")
