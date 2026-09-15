"""
============================================================
Tomato Dataset - CNN Model Comparison
============================================================

Models:
    1. DenseNet121
    2. ResNet50
    3. VGG16

Default classification:
    clear vs defected

Dataset structure:

Tomato Dataset/
│
├── Manualy Segmented/
├── Unsegmented/j
├── Dataset anotations.xls
└── train_compare.py

The program will:

1. Read the Excel annotation file
2. Find images from both image folders
3. Match image names with Excel annotations
4. Split data by Tomato ID to prevent data leakage
5. Train DenseNet121
6. Train ResNet50
7. Train VGG16
8. Calculate:
       Accuracy
       Precision
       Recall
       F1-score
       Loss
       Number of parameters
       Training time
9. Save best model for each architecture
10. Save confusion matrices
11. Save training curves
12. Save everything into comparison.xlsx

============================================================
INSTALL
============================================================

Run this in VS Code terminal:

pip install torch torchvision pandas xlrd scikit-learn pillow matplotlib openpyxl

Then:

python train_compare.py

============================================================
"""

from pathlib import Path
import random
import time
import copy
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Automatically uses the folder where this Python file is located.
DATASET_ROOT = Path(__file__).resolve().parent

ANNOTATION_FILE = DATASET_ROOT / "Dataset_anotations.xls"

IMAGE_FOLDERS = [
    DATASET_ROOT / "Manualy Segmented",
    DATASET_ROOT / "Unsegmented"
]

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

# "side_case" = clear vs defected
# "ripeness"  = ripe vs unripe

TARGET = "side_case"


# ------------------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = DATASET_ROOT / "model_comparison_results"

MODEL_DIR = OUTPUT_DIR / "models"

GRAPH_DIR = OUTPUT_DIR / "graphs"

MATRIX_DIR = OUTPUT_DIR / "confusion_matrices"

SPLIT_DIR = OUTPUT_DIR / "splits"


# ============================================================
# 2. CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)

GRAPH_DIR.mkdir(exist_ok=True)

MATRIX_DIR.mkdir(exist_ok=True)

SPLIT_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)

print("TOMATO DATASET - MODEL COMPARISON")

print("=" * 75)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# 5. CHECK DATASET
# ============================================================

if not ANNOTATION_FILE.exists():

    raise FileNotFoundError(
        f"\nExcel file not found:\n{ANNOTATION_FILE}\n\n"
        "Make sure Dataset anotations.xls is in the same folder "
        "as this Python file."
    )


# ============================================================
# 6. READ EXCEL
# ============================================================

print("Reading Excel annotations...")

try:

    excel = pd.ExcelFile(
        ANNOTATION_FILE,
        engine="xlrd"
    )

except Exception as e:

    raise RuntimeError(
        "\nCould not read .xls file.\n"
        "Install xlrd using:\n\n"
        "pip install xlrd\n\n"
        f"Original error:\n{e}"
    )


print("Excel sheets:")

print(excel.sheet_names)


# ------------------------------------------------------------
# Automatically find the sheet containing "Name"
# ------------------------------------------------------------

annotation_df = None

selected_sheet = None

for sheet in excel.sheet_names:

    try:

        temp = pd.read_excel(
            ANNOTATION_FILE,
            sheet_name=sheet,
            header=None,
            engine="xlrd"
        )

        found_header = False

        for row_index in range(
            min(15, len(temp))
        ):

            row_values = (
                temp.iloc[row_index]
                .astype(str)
                .str.strip()
                .tolist()
            )

            if "Name" in row_values:

                annotation_df = pd.read_excel(
                    ANNOTATION_FILE,
                    sheet_name=sheet,
                    header=row_index,
                    engine="xlrd"
                )

                selected_sheet = sheet

                found_header = True

                break

        if found_header:

            break

    except Exception:

        continue


if annotation_df is None:

    raise ValueError(
        "\nCould not find an Excel sheet containing "
        "the 'Name' column."
    )


df = annotation_df.copy()

df.columns = [
    str(c).strip()
    for c in df.columns
]


print(
    f"\nUsing sheet: {selected_sheet}"
)

print(
    "\nColumns found:"
)

for column in df.columns:

    print(
        "  ",
        column
    )


# ============================================================
# 7. FIND REQUIRED COLUMNS
# ============================================================

def find_column(possible_names):

    for column in df.columns:

        clean = (
            str(column)
            .strip()
            .lower()
        )

        for name in possible_names:

            if clean == name.lower():

                return column

    return None


name_column = find_column([
    "Name"
])

ripeness_column = find_column([
    "Ripeness"
])

side_case_column = find_column([
    "Side Case",
    "SideCase",
    "Side_Case"
])


if name_column is None:

    raise ValueError(
        "\nCould not find the 'Name' column."
    )


if TARGET == "side_case":

    if side_case_column is None:

        raise ValueError(
            "\nCould not find 'Side Case' column."
        )

    target_column = side_case_column

    class_names = [
        "clear",
        "defected"
    ]

elif TARGET == "ripeness":

    if ripeness_column is None:

        raise ValueError(
            "\nCould not find 'Ripeness' column."
        )

    target_column = ripeness_column

    class_names = [
        "ripe",
        "unripe"
    ]

else:

    raise ValueError(
        "TARGET must be 'side_case' or 'ripeness'"
    )


# ============================================================
# 8. CLEAN LABELS
# ============================================================

df["image_name"] = (
    df[name_column]
    .astype(str)
    .str.strip()
)


df["label"] = (
    df[target_column]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Remove invalid values

df = df[
    df["label"].isin(class_names)
].copy()


# ============================================================
# 9. EXTRACT TOMATO ID
# ============================================================

# Examples:
#
# T0001_E  -> T0001
# T0001_S1 -> T0001
# T0001_S2 -> T0001
# T0001_T  -> T0001

df["tomato_id"] = (
    df["image_name"]
    .str.extract(
        r"^(T\d+)",
        expand=False
    )
)


df = df.dropna(
    subset=["tomato_id"]
)


print("\nNumber of annotated images:")

print(
    len(df)
)


print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# 10. FIND IMAGES
# ============================================================

print("\nSearching image folders...")

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


image_records = []


for folder in IMAGE_FOLDERS:

    if not folder.exists():

        print(
            f"WARNING: Folder not found: {folder}"
        )

        continue


    print(
        f"Searching: {folder}"
    )


    for file in folder.rglob("*"):

        if not file.is_file():

            continue


        if file.suffix.lower() not in valid_extensions:

            continue


        image_records.append({
            "image_name": file.stem.strip(),
            "image_path": str(file),
            "source_folder": folder.name
        })


image_df = pd.DataFrame(
    image_records
)


if len(image_df) == 0:

    raise RuntimeError(
        "\nNo images found.\n"
        "Check your folder paths."
    )


print(
    f"\nTotal image files found: "
    f"{len(image_df)}"
)


# ============================================================
# 11. MATCH EXCEL WITH IMAGES
# ============================================================

merged = df.merge(
    image_df,
    on="image_name",
    how="inner"
)


# Avoid duplicate exact paths

merged = merged.drop_duplicates(
    subset=["image_path"]
).copy()


print(
    f"Usable image files: "
    f"{len(merged)}"
)


if len(merged) == 0:

    raise RuntimeError(
        "\nNo Excel annotations matched image filenames.\n"
        "Check the names in the Excel file and image files."
    )


# ============================================================
# 12. DATASET INFORMATION
# ============================================================

print("\nFinal dataset:")

print(
    merged[
        ["image_name", "tomato_id", "label"]
    ].head()
)


print("\nFinal class distribution:")

print(
    merged["label"].value_counts()
)


# ============================================================
# 13. GROUPED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print(
    "\nCreating tomato-level train/validation/test split..."
)


tomato_ids = (
    merged["tomato_id"]
    .unique()
)


train_ids, temporary_ids = train_test_split(
    tomato_ids,
    test_size=(
        VAL_RATIO + TEST_RATIO
    ),
    random_state=SEED
)


test_ratio_relative = (
    TEST_RATIO /
    (VAL_RATIO + TEST_RATIO)
)


val_ids, test_ids = train_test_split(
    temporary_ids,
    test_size=test_ratio_relative,
    random_state=SEED
)


train_df = merged[
    merged["tomato_id"].isin(train_ids)
].copy()


val_df = merged[
    merged["tomato_id"].isin(val_ids)
].copy()


test_df = merged[
    merged["tomato_id"].isin(test_ids)
].copy()


train_df["split"] = "train"

val_df["split"] = "validation"

test_df["split"] = "test"


print("\nDataset split:")

print(
    f"Train: "
    f"{len(train_df)} images / "
    f"{len(train_ids)} tomatoes"
)

print(
    f"Validation: "
    f"{len(val_df)} images / "
    f"{len(val_ids)} tomatoes"
)

print(
    f"Test: "
    f"{len(test_df)} images / "
    f"{len(test_ids)} tomatoes"
)


# Save splits

train_df.to_csv(
    SPLIT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    SPLIT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    SPLIT_DIR / "test.csv",
    index=False
)


# ============================================================
# 14. LABEL ENCODING
# ============================================================

label_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


# ============================================================
# 15. TRANSFORMS
# ============================================================

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]


train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    )
])


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    )
])


# ============================================================
# 16. PYTORCH DATASET
# ============================================================

class TomatoDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index
    ):

        row = self.dataframe.iloc[index]


        image_path = Path(
            row["image_path"]
        )


        image = Image.open(
            image_path
        ).convert("RGB")


        label = label_to_index[
            row["label"]
        ]


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# 17. DATA LOADERS
# ============================================================

train_dataset = TomatoDataset(
    train_df,
    train_transform
)


val_dataset = TomatoDataset(
    val_df,
    test_transform
)


test_dataset = TomatoDataset(
    test_df,
    test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ============================================================
# 18. MODEL CREATION
# ============================================================

def create_model(model_name):

    if model_name == "DenseNet121":

        weights = (
            models.DenseNet121_Weights.DEFAULT
        )

        model = models.densenet121(
            weights=weights
        )

        input_features = (
            model.classifier.in_features
        )

        model.classifier = nn.Linear(
            input_features,
            len(class_names)
        )


    elif model_name == "ResNet50":

        weights = (
            models.ResNet50_Weights.DEFAULT
        )

        model = models.resnet50(
            weights=weights
        )

        input_features = (
            model.fc.in_features
        )

        model.fc = nn.Linear(
            input_features,
            len(class_names)
        )


    elif model_name == "VGG16":

        weights = (
            models.VGG16_Weights.DEFAULT
        )

        model = models.vgg16(
            weights=weights
        )

        input_features = (
            model.classifier[-1]
            .in_features
        )

        model.classifier[-1] = nn.Linear(
            input_features,
            len(class_names)
        )


    else:

        raise ValueError(
            "Unknown model"
        )


    return model.to(DEVICE)


# ============================================================
# 19. CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["label"]
    .value_counts()
)


class_weights = []


for class_name in class_names:

    count = class_counts.get(
        class_name,
        1
    )

    weight = (
        len(train_df) /
        (
            len(class_names)
            * count
        )
    )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 20. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()


    total_loss = 0

    all_labels = []

    all_predictions = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            total_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = (
                outputs
                .argmax(dim=1)
            )


            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )


            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )


    average_loss = (
        total_loss /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return (
        average_loss,
        accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# 21. TRAIN ONE MODEL
# ============================================================

def train_model(
    model_name
):

    print("\n")

    print(
        "=" * 75
    )

    print(
        f"TRAINING: {model_name}"
    )

    print(
        "=" * 75
    )


    model = create_model(
        model_name
    )


    # Number of parameters

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=2
        )
    )


    best_val_loss = float(
        "inf"
    )


    best_model_state = None

    best_epoch = 0


    history = []


    start_time = time.time()


    for epoch in range(
        1,
        EPOCHS + 1
    ):


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()


        running_loss = 0

        train_labels = []

        train_predictions = []


        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            optimizer.zero_grad()


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = (
                outputs
                .argmax(dim=1)
            )


            train_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )


            train_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )


        train_loss = (
            running_loss /
            len(train_dataset)
        )


        train_accuracy = (
            accuracy_score(
                train_labels,
                train_predictions
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy, _, _ = (
            evaluate_model(
                model,
                val_loader
            )
        )


        scheduler.step(
            val_loss
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        history.append({

            "Model": model_name,

            "Epoch": epoch,

            "Train Loss": train_loss,

            "Train Accuracy": train_accuracy,

            "Validation Loss": val_loss,

            "Validation Accuracy": val_accuracy,

            "Learning Rate": current_lr

        })


        print(

            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "

            f"Train Loss: "
            f"{train_loss:.4f} | "

            f"Train Acc: "
            f"{train_accuracy:.4f} | "

            f"Val Loss: "
            f"{val_loss:.4f} | "

            f"Val Acc: "
            f"{val_accuracy:.4f}"

        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )


            model_path = (
                MODEL_DIR /
                f"{model_name}_best.pth"
            )


            torch.save({

                "model_name":
                    model_name,

                "model_state_dict":
                    best_model_state,

                "class_names":
                    class_names,

                "target":
                    TARGET,

                "image_size":
                    IMAGE_SIZE,

                "best_epoch":
                    best_epoch,

                "validation_loss":
                    best_val_loss

            }, model_path)


            print(
                "  -> Best model saved."
            )


    training_time = (
        time.time()
        - start_time
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_model_state
    )


    # ========================================================
    # TEST
    # ========================================================

    test_loss, test_accuracy, y_true, y_pred = (
        evaluate_model(
            model,
            test_loader
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report_dict = classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        output_dict=True,

        zero_division=0
    )


    report_text = classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        digits=4,

        zero_division=0
    )


    print("\n")

    print(
        f"========== {model_name} TEST RESULTS =========="
    )


    print(
        f"Test Loss:     {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    print(
        f"Precision:     {precision:.4f}"
    )

    print(
        f"Recall:        {recall:.4f}"
    )

    print(
        f"F1 Score:      {f1:.4f}"
    )

    print(
        f"Training Time: "
        f"{training_time / 60:.2f} minutes"
    )


    print("\nClassification report:")

    print(
        report_text
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(
            len(class_names)
        )
    )


    # Save confusion matrix as Excel

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual_{c}"
            for c in class_names
        ],
        columns=[
            f"Predicted_{c}"
            for c in class_names
        ]
    )


    # Save confusion matrix PNG

    plt.figure(
        figsize=(7, 6)
    )


    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        f"{model_name} - Confusion Matrix"
    )


    plt.colorbar()


    tick_marks = np.arange(
        len(class_names)
    )


    plt.xticks(
        tick_marks,
        class_names,
        rotation=45
    )


    plt.yticks(
        tick_marks,
        class_names
    )


    threshold = (
        cm.max() / 2
        if cm.size > 0
        else 0
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(

                j,

                i,

                str(cm[i, j]),

                horizontalalignment="center",

                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )


    plt.ylabel(
        "Actual"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.tight_layout()


    plt.savefig(
        MATRIX_DIR /
        f"{model_name}_confusion_matrix.png",
        dpi=300
    )


    plt.close()


    # ========================================================
    # TRAINING CURVES
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    # Accuracy graph

    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(

        history_df["Epoch"],

        history_df["Train Accuracy"],

        label="Train Accuracy"
    )


    plt.plot(

        history_df["Epoch"],

        history_df["Validation Accuracy"],

        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        f"{model_name} - Accuracy"
    )


    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        GRAPH_DIR /
        f"{model_name}_accuracy.png",

        dpi=300
    )


    plt.close()


    # Loss graph

    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(

        history_df["Epoch"],

        history_df["Train Loss"],

        label="Train Loss"
    )


    plt.plot(

        history_df["Epoch"],

        history_df["Validation Loss"],

        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.title(
        f"{model_name} - Loss"
    )


    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        GRAPH_DIR /
        f"{model_name}_loss.png",

        dpi=300
    )


    plt.close()


    # Save individual history

    history_df.to_csv(

        OUTPUT_DIR /
        f"{model_name}_history.csv",

        index=False
    )


    # Save text report

    with open(

        OUTPUT_DIR /
        f"{model_name}_classification_report.txt",

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            report_text
        )


    # ========================================================
    # PER CLASS RESULTS
    # ========================================================

    per_class_results = []


    for class_name in class_names:

        metrics = report_dict[
            class_name
        ]


        per_class_results.append({

            "Model":
                model_name,

            "Class":
                class_name,

            "Precision":
                metrics["precision"],

            "Recall":
                metrics["recall"],

            "F1 Score":
                metrics["f1-score"],

            "Support":
                metrics["support"]

        })


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    overall_result = {

        "Model":
            model_name,

        "Dataset Images":
            len(merged),

        "Train Images":
            len(train_df),

        "Validation Images":
            len(val_df),

        "Test Images":
            len(test_df),

        "Total Parameters":
            total_parameters,

        "Trainable Parameters":
            trainable_parameters,

        "Best Epoch":
            best_epoch,

        "Best Validation Loss":
            best_val_loss,

        "Test Loss":
            test_loss,

        "Test Accuracy":
            test_accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "Training Time (seconds)":
            training_time,

        "Training Time (minutes)":
            training_time / 60

    }


    return (
        overall_result,
        per_class_results,
        cm_df,
        history_df
    )


# ============================================================
# 22. TRAIN ALL THREE MODELS
# ============================================================

MODELS = [

    "DenseNet121",

    "ResNet50",

    "VGG16"

]


all_results = []

all_class_results = []

all_confusion_matrices = []

all_histories = []


for model_name in MODELS:

    result, class_results, cm_df, history_df = (

        train_model(
            model_name
        )

    )


    all_results.append(
        result
    )


    all_class_results.extend(
        class_results
    )


    cm_df.insert(
        0,
        "Model",
        model_name
    )


    all_confusion_matrices.append(
        cm_df
    )


    all_histories.append(
        history_df
    )


# ============================================================
# 23. CREATE FINAL DATAFRAMES
# ============================================================

comparison_df = pd.DataFrame(
    all_results
)


class_metrics_df = pd.DataFrame(
    all_class_results
)


confusion_df = pd.concat(
    all_confusion_matrices,
    ignore_index=True
)


history_df = pd.concat(
    all_histories,
    ignore_index=True
)


# ============================================================
# 24. SORT BY TEST ACCURACY
# ============================================================

comparison_df = comparison_df.sort_values(

    by="Test Accuracy",

    ascending=False

).reset_index(
    drop=True
)


comparison_df.insert(

    0,

    "Rank",

    range(
        1,
        len(comparison_df) + 1
    )

)


# ============================================================
# 25. SAVE EXCEL FILE
# ============================================================

excel_output = (
    OUTPUT_DIR /
    "comparison.xlsx"
)


with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:


    # --------------------------------------------------------
    # Sheet 1
    # --------------------------------------------------------

    comparison_df.to_excel(

        writer,

        sheet_name="Model Comparison",

        index=False
    )


    # --------------------------------------------------------
    # Sheet 2
    # --------------------------------------------------------

    class_metrics_df.to_excel(

        writer,

        sheet_name="Per Class Metrics",

        index=False
    )


    # --------------------------------------------------------
    # Sheet 3
    # --------------------------------------------------------

    confusion_df.to_excel(

        writer,

        sheet_name="Confusion Matrices",

        index=False
    )


    # --------------------------------------------------------
    # Sheet 4
    # --------------------------------------------------------

    history_df.to_excel(

        writer,

        sheet_name="Training History",

        index=False
    )


    # --------------------------------------------------------
    # Sheet 5 - Dataset Split
    # --------------------------------------------------------

    split_df = pd.concat(

        [
            train_df,
            val_df,
            test_df
        ],

        ignore_index=True
    )


    split_columns = [

        "image_name",

        "tomato_id",

        "source_folder",

        "label",

        "split",

        "image_path"

    ]


    split_columns = [

        c

        for c in split_columns

        if c in split_df.columns

    ]


    split_df[
        split_columns
    ].to_excel(

        writer,

        sheet_name="Dataset Split",

        index=False
    )


    # --------------------------------------------------------
    # Sheet 6 - Configuration
    # --------------------------------------------------------

    config_df = pd.DataFrame({

        "Setting": [

            "Target",

            "Classes",

            "Image Size",

            "Batch Size",

            "Epochs",

            "Learning Rate",

            "Weight Decay",

            "Train Ratio",

            "Validation Ratio",

            "Test Ratio",

            "Random Seed",

            "Device"

        ],

        "Value": [

            TARGET,

            ", ".join(
                class_names
            ),

            IMAGE_SIZE,

            BATCH_SIZE,

            EPOCHS,

            LEARNING_RATE,

            WEIGHT_DECAY,

            TRAIN_RATIO,

            VAL_RATIO,

            TEST_RATIO,

            SEED,

            str(DEVICE)

        ]

    })


    config_df.to_excel(

        writer,

        sheet_name="Configuration",

        index=False
    )


print("\n")

print(
    "=" * 75
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "=" * 75
)


print(

    comparison_df[
        [
            "Rank",

            "Model",

            "Test Accuracy",

            "Precision",

            "Recall",

            "F1 Score",

            "Training Time (minutes)"

        ]

    ].to_string(
        index=False
    )

)


# ============================================================
# 26. CREATE MODEL COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.bar(

    comparison_df["Model"],

    comparison_df["Test Accuracy"]

)


plt.xlabel(
    "Model"
)


plt.ylabel(
    "Test Accuracy"
)


plt.title(
    "Comparison of CNN Models"
)


plt.ylim(
    0,
    1
)


for i, value in enumerate(

    comparison_df[
        "Test Accuracy"
    ]

):

    plt.text(

        i,

        value + 0.02,

        f"{value:.3f}",

        ha="center"

    )


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    GRAPH_DIR /
    "model_comparison_accuracy.png",

    dpi=300
)


plt.close()


# ============================================================
# 27. F1 SCORE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.bar(

    comparison_df["Model"],

    comparison_df["F1 Score"]

)


plt.xlabel(
    "Model"
)


plt.ylabel(
    "F1 Score"
)


plt.title(
    "F1 Score Comparison"
)


plt.ylim(
    0,
    1
)


for i, value in enumerate(

    comparison_df[
        "F1 Score"
    ]

):

    plt.text(

        i,

        value + 0.02,

        f"{value:.3f}",

        ha="center"

    )


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    GRAPH_DIR /
    "model_comparison_f1.png",

    dpi=300
)


plt.close()


# ============================================================
# 28. SAVE JSON CONFIG
# ============================================================

config = {

    "dataset_root":
        str(DATASET_ROOT),

    "annotation_file":
        str(ANNOTATION_FILE),

    "target":
        TARGET,

    "classes":
        class_names,

    "models":
        MODELS,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "train_ratio":
        TRAIN_RATIO,

    "validation_ratio":
        VAL_RATIO,

    "test_ratio":
        TEST_RATIO,

    "seed":
        SEED,

    "device":
        str(DEVICE)

}


with open(

    OUTPUT_DIR /
    "config.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        config,

        f,

        indent=4

    )


# ============================================================
# 29. FINISHED
# ============================================================

print("\n")

print(
    "=" * 75
)

print(
    "TRAINING AND COMPARISON FINISHED"
)

print(
    "=" * 75
)


print(
    "\nResults folder:"
)

print(
    OUTPUT_DIR
)


print(
    "\nImportant files:"
)

print(
    "  comparison.xlsx"
)

print(
    "  models/DenseNet121_best.pth"
)

print(
    "  models/ResNet50_best.pth"
)

print(
    "  models/VGG16_best.pth"
)

print(
    "  graphs/model_comparison_accuracy.png"
)

print(
    "  graphs/model_comparison_f1.png"
)

print(
    "  confusion_matrices/"
)

print(
    "  splits/"
)

print(
    "\nDone!"
)