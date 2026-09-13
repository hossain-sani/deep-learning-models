import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights

from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================
# 1. Device
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================
# 2. Dataset paths
# ============================================

train_dir = "train"
valid_dir = "valid"


# ============================================
# 3. Image transformations
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================
# 4. Load datasets
# ============================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    valid_dir,
    transform=valid_transform
)


# ============================================
# 5. Data loaders
# ============================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================
# 6. Print dataset information
# ============================================

print("\nClasses:")
print(train_dataset.classes)

print("\nNumber of classes:", len(train_dataset.classes))
print("Training images:", len(train_dataset))
print("Validation images:", len(valid_dataset))


# ============================================
# 7. Load DenseNet121
# ============================================

weights = DenseNet121_Weights.DEFAULT

model = densenet121(weights=weights)


# ============================================
# 8. Replace final classification layer
# ============================================

num_features = model.classifier.in_features

model.classifier = nn.Linear(
    num_features,
    len(train_dataset.classes)
)


model = model.to(device)


# ============================================
# 9. Loss function
# ============================================

criterion = nn.CrossEntropyLoss()


# ============================================
# 10. Optimizer
# ============================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ============================================
# 11. Training
# ============================================

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs}"
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=f"{accuracy:.2f}%"
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total


    # ========================================
    # Validation
    # ========================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()


    val_accuracy = 100 * val_correct / val_total
    val_loss = val_loss / len(valid_loader)


    print(
        f"\nEpoch [{epoch + 1}/{num_epochs}]"
        f"\nTrain Loss: {epoch_loss:.4f}"
        f"\nTrain Accuracy: {epoch_accuracy:.2f}%"
        f"\nValidation Loss: {val_loss:.4f}"
        f"\nValidation Accuracy: {val_accuracy:.2f}%"
    )


# ============================================
# 12. Save model
# ============================================

torch.save(
    model.state_dict(),
    "densenet121_tomato.pth"
)

print("\nModel saved successfully!")