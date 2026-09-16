"""
Train and compare DenseNet121, ResNet50, and VGG16 in one run.

Run:
    python compare_all_models.py

Example:
    python compare_all_models.py --epochs 15 --batch-size 16 --target side_case

The script creates model_comparison_results_one_file/ containing:
    comparison.xlsx
    comparison.csv
    config.json
    splits/
    models/
    graphs/
    confusion_matrices/
    *_classification_report.txt
    *_history.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


MODELS_TO_COMPARE = ("DenseNet121", "ResNet50", "VGG16")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def percent(part, total):
    return 100.0 * part / total if total else 0.0


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        self.streams[0].write(text)
        self.streams[0].flush()
        for stream in self.streams[1:]:
            stream.write(text.replace("\r", "\n"))
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root)
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=root / "model_comparison_results_one_file")
    parser.add_argument("--log-file", type=str, default="terminal_output.txt")
    parser.add_argument("--target", choices=("side_case", "ripeness"), default="side_case")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_column(columns, names):
    names = {name.lower().replace("_", " ") for name in names}
    for column in columns:
        normalized = str(column).strip().lower().replace("_", " ")
        if normalized in names:
            return column
    return None


def read_annotations(annotation_file: Path, target: str):
    excel = pd.ExcelFile(annotation_file, engine="xlrd")
    annotation_df = None
    selected_sheet = None

    for sheet in excel.sheet_names:
        raw = pd.read_excel(annotation_file, sheet_name=sheet, header=None, engine="xlrd")
        for row_index in range(min(15, len(raw))):
            values = raw.iloc[row_index].astype(str).str.strip().tolist()
            if "Name" in values:
                annotation_df = pd.read_excel(
                    annotation_file, sheet_name=sheet, header=row_index, engine="xlrd"
                )
                selected_sheet = sheet
                break
        if annotation_df is not None:
            break

    if annotation_df is None:
        raise ValueError("Could not find a worksheet containing a 'Name' column.")

    annotation_df.columns = [str(column).strip() for column in annotation_df.columns]
    name_column = find_column(annotation_df.columns, ("Name",))
    target_column = find_column(
        annotation_df.columns,
        ("Side Case", "SideCase") if target == "side_case" else ("Ripeness",),
    )
    class_names = ["clear", "defected"] if target == "side_case" else ["ripe", "unripe"]

    if name_column is None or target_column is None:
        raise ValueError(
            f"Required columns were not found. Available columns: {list(annotation_df.columns)}"
        )

    result = annotation_df.copy()
    result["image_name"] = result[name_column].astype(str).str.strip()
    result["label"] = result[target_column].astype(str).str.strip().str.lower()
    result = result[result["label"].isin(class_names)].copy()
    result["tomato_id"] = result["image_name"].str.extract(r"^(T\d+)", expand=False)
    result = result.dropna(subset=["tomato_id"])

    print(f"Using sheet: {selected_sheet}")
    print(f"Annotated rows for {target}: {len(result)}")
    label_counts = result["label"].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({percent(count, len(result)):.2f}%)")
    return result, class_names


def find_images(image_folders):
    records = []
    for folder in image_folders:
        if not folder.exists():
            print(f"Warning: image folder not found: {folder}")
            continue
        for image_path in folder.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                records.append(
                    {
                        "image_name": image_path.stem.strip(),
                        "image_path": str(image_path),
                        "source_folder": folder.name,
                    }
                )
    if not records:
        raise RuntimeError("No image files were found in the configured image folders.")
    return pd.DataFrame(records)


def make_splits(merged, args, split_dir):
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0")

    tomato_ids = merged["tomato_id"].unique()
    train_ids, temporary_ids = train_test_split(
        tomato_ids,
        test_size=args.val_ratio + args.test_ratio,
        random_state=args.seed,
    )
    relative_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_ids, test_ids = train_test_split(
        temporary_ids, test_size=relative_test_ratio, random_state=args.seed
    )

    train_df = merged[merged["tomato_id"].isin(train_ids)].copy()
    val_df = merged[merged["tomato_id"].isin(val_ids)].copy()
    test_df = merged[merged["tomato_id"].isin(test_ids)].copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "validation.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    return train_df, val_df, test_df


class TomatoDataset(Dataset):
    def __init__(self, dataframe, label_to_index, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        with Image.open(Path(row["image_path"])) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(
            self.label_to_index[row["label"]], dtype=torch.long
        )


def create_model(model_name, class_count, device, pretrained):
    try:
        if model_name == "DenseNet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
            model.classifier = nn.Linear(model.classifier.in_features, class_count)
        elif model_name == "ResNet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, class_count)
        elif model_name == "VGG16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as error:
        if not pretrained:
            raise
        print(f"Could not load pretrained weights for {model_name}: {error}")
        print("Continuing with randomly initialized weights.")
        return create_model(model_name, class_count, device, pretrained=False)
    return model.to(device)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels, predictions = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            outputs = model(images)
            total_loss += criterion(outputs, batch_labels).item() * images.size(0)
            labels.extend(batch_labels.cpu().numpy())
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        accuracy_score(labels, predictions),
        labels,
        predictions,
    )


def save_confusion_matrix(cm, class_names, model_name, output_path):
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(cm, interpolation="nearest", cmap="Blues")
    axis.set_title(f"{model_name} - Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45)
    axis.set_yticks(range(len(class_names)), class_names)
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            axis.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                color="white" if cm[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def train_one_model(
    model_name, train_loader, val_loader, test_loader, criterion, class_names,
    train_size, args, device, output_dirs,
):
    set_seed(args.seed)
    model = create_model(model_name, len(class_names), device, not args.no_pretrained)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    start_time = time.time()

    total_batches = len(train_loader)
    print(f"\n{'=' * 72}\nTraining {model_name}\n{'=' * 72}")
    print(f"Model progress is shown as {model_name}; {args.epochs} epochs; {total_batches} batches per epoch")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_labels, train_predictions = [], []
        for batch_number, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_labels.extend(labels.detach().cpu().numpy())
            train_predictions.extend(outputs.detach().argmax(dim=1).cpu().numpy())
            print(
                f"\r  Epoch {epoch:02d}/{args.epochs} "
                f"({percent(epoch, args.epochs):6.2f}% total) | "
                f"batch {batch_number:03d}/{total_batches} "
                f"({percent(batch_number, total_batches):6.2f}%)",
                end="",
                flush=True,
            )

        train_loss = running_loss / train_size
        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "Model": model_name,
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_accuracy,
                "Learning Rate": current_lr,
            }
        )
        print(
            f"\rEpoch {epoch:02d}/{args.epochs} ({percent(epoch, args.epochs):.2f}%) | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.4f} | val loss {val_loss:.4f} | val acc {val_accuracy:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": best_state,
                    "class_names": class_names,
                    "target": args.target,
                    "image_size": args.image_size,
                    "best_epoch": best_epoch,
                    "validation_loss": best_loss,
                },
                output_dirs["models"] / f"{model_name}_best.pth",
            )

    model.load_state_dict(best_state)
    test_loss, test_accuracy, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dirs["root"] / f"{model_name}_history.csv", index=False)
    (output_dirs["root"] / f"{model_name}_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    save_confusion_matrix(
        cm, class_names, model_name,
        output_dirs["matrices"] / f"{model_name}_confusion_matrix.png",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["Epoch"], history_df["Train Accuracy"], label="Train")
    axes[0].plot(history_df["Epoch"], history_df["Validation Accuracy"], label="Validation")
    axes[0].set_title(f"{model_name} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history_df["Epoch"], history_df["Train Loss"], label="Train")
    axes[1].plot(history_df["Epoch"], history_df["Validation Loss"], label="Validation")
    axes[1].set_title(f"{model_name} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dirs["graphs"] / f"{model_name}_training_curves.png", dpi=300)
    plt.close(figure)

    elapsed = time.time() - start_time
    result = {
        "Model": model_name,
        "Total Parameters": total_parameters,
        "Trainable Parameters": trainable_parameters,
        "Best Epoch": best_epoch,
        "Best Validation Loss": best_loss,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (minutes)": elapsed / 60,
    }
    per_class = [
        {
            "Model": model_name,
            "Class": class_name,
            "Precision": report_dict[class_name]["precision"],
            "Recall": report_dict[class_name]["recall"],
            "F1 Score": report_dict[class_name]["f1-score"],
            "Support": report_dict[class_name]["support"],
        }
        for class_name in class_names
    ]
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{name}" for name in class_names],
        columns=[f"Predicted_{name}" for name in class_names],
    ).reset_index(names="Actual")
    cm_df.insert(0, "Model", model_name)
    print(
        f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%) | "
        f"precision: {precision:.4f} ({precision * 100:.2f}%) | "
        f"recall: {recall:.4f} ({recall * 100:.2f}%) | "
        f"weighted F1: {f1:.4f} ({f1 * 100:.2f}%)"
    )
    return result, per_class, cm_df, history_df


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.annotation_file is None:
        args.annotation_file = args.dataset_root / "Dataset_anotations.xls"
    args.dataset_root = args.dataset_root.resolve()
    args.annotation_file = args.annotation_file.resolve()
    args.output_dir = args.output_dir.resolve()
    output_dirs = {
        "root": args.output_dir,
        "models": args.output_dir / "models",
        "graphs": args.output_dir / "graphs",
        "matrices": args.output_dir / "confusion_matrices",
        "splits": args.output_dir / "splits",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    if not args.annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.annotation_file}")

    log_path = args.output_dir / args.log_file
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Terminal output is being saved to: {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Target: {args.target}")

    annotations, class_names = read_annotations(args.annotation_file, args.target)
    image_df = find_images(
        [args.dataset_root / "Manualy Segmented", args.dataset_root / "Unsegmented"]
    )
    merged = annotations.merge(image_df, on="image_name", how="inner")
    merged = merged.drop_duplicates(subset=["image_path"]).copy()
    if merged.empty:
        raise RuntimeError("No annotation rows matched the image filenames.")
    train_df, val_df, test_df = make_splits(merged, args, output_dirs["splits"])
    print("\nDataset split percentages:")
    print(f"  Train:      {len(train_df):5d} ({percent(len(train_df), len(merged)):6.2f}%)")
    print(f"  Validation: {len(val_df):5d} ({percent(len(val_df), len(merged)):6.2f}%)")
    print(f"  Test:       {len(test_df):5d} ({percent(len(test_df), len(merged)):6.2f}%)")

    label_to_index = {name: index for index, name in enumerate(class_names)}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_loader = DataLoader(
        TomatoDataset(train_df, label_to_index, train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TomatoDataset(val_df, label_to_index, test_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TomatoDataset(test_df, label_to_index, test_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    counts = train_df["label"].value_counts()
    print("\nTraining class percentages:")
    for class_name in class_names:
        count = counts.get(class_name, 0)
        print(f"  {class_name}: {count} ({percent(count, len(train_df)):.2f}%)")
    class_weights = torch.tensor(
        [len(train_df) / (len(class_names) * counts.get(name, 1)) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    results, per_class_results, confusion_matrices, histories = [], [], [], []
    total_models = len(MODELS_TO_COMPARE)
    for model_number, model_name in enumerate(MODELS_TO_COMPARE, start=1):
        print(
            f"\n\nOVERALL MODEL PROGRESS: {model_number}/{total_models} "
            f"({percent(model_number, total_models):.2f}%) - {model_name}"
        )
        result, per_class, cm_df, history_df = train_one_model(
            model_name, train_loader, val_loader, test_loader, criterion, class_names,
            len(train_df), args, device, output_dirs,
        )
        result.update({
            "Dataset Images": len(merged),
            "Train Images": len(train_df),
            "Train Percentage": percent(len(train_df), len(merged)),
            "Validation Images": len(val_df),
            "Validation Percentage": percent(len(val_df), len(merged)),
            "Test Images": len(test_df),
            "Test Percentage": percent(len(test_df), len(merged)),
        })
        results.append(result)
        per_class_results.extend(per_class)
        confusion_matrices.append(cm_df)
        histories.append(history_df)

    comparison_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))
    class_metrics_df = pd.DataFrame(per_class_results)
    confusion_df = pd.concat(confusion_matrices, ignore_index=True)
    history_df = pd.concat(histories, ignore_index=True)
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    comparison_df.to_csv(args.output_dir / "comparison.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "comparison.xlsx", engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="Model Comparison", index=False)
        class_metrics_df.to_excel(writer, sheet_name="Per Class Metrics", index=False)
        confusion_df.to_excel(writer, sheet_name="Confusion Matrices", index=False)
        history_df.to_excel(writer, sheet_name="Training History", index=False)
        split_df.to_excel(writer, sheet_name="Dataset Split", index=False)

    config = vars(args).copy()
    config.update({
        "dataset_root": str(args.dataset_root),
        "annotation_file": str(args.annotation_file),
        "output_dir": str(args.output_dir),
        "models": list(MODELS_TO_COMPARE),
        "classes": class_names,
        "device": str(device),
    })
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print("\nFinal comparison:")
    print(comparison_df[["Rank", "Model", "Test Accuracy", "Precision", "Recall", "F1 Score"]].to_string(index=False))
    print(f"\nSaved all comparison files to: {args.output_dir}")
    print(f"Saved terminal output to: {log_path}")
    log_file.close()


if __name__ == "__main__":
    main()
