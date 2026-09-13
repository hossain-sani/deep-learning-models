import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(v):
    """Converts a string representation to boolean in argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_subset(dataset, fraction, seed=42):
    """Returns a deterministic subset of the dataset based on the fraction."""
    if fraction >= 1.0:
        return dataset
    
    num_samples = int(len(dataset) * fraction)
    if num_samples <= 0:
        num_samples = min(len(dataset), 1)
        
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    return Subset(dataset, indices)


# Define standard transforms for DenseNet121 (using ImageNet normalization)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    """Trains the model and returns loss/accuracy history."""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\nStarting training on device: {device}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # === Training Phase ===
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data).item()
            total += images.size(0)
            
            # Show running performance
            pbar.set_postfix(loss=loss.item(), acc=corrects/total)
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = corrects / total
        
        # === Validation Phase ===
        model.eval()
        val_running_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        with torch.no_grad():
            vpbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in vpbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += images.size(0)
                
                vpbar.set_postfix(loss=loss.item(), acc=val_corrects/val_total)
                
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_corrects / val_total
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs} summary: "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
              
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"==> Saved new best checkpoint to {save_path}")
            
    print(f"\nTraining completed. Best Validation Accuracy: {best_val_acc:.4f} (Loss: {best_val_loss:.4f})")
    return history


def plot_curves(history, output_path="learning_curves.png"):
    """Generates and saves the loss and accuracy learning curves."""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Learning curves plot saved to {output_path}")


def evaluate_model(model, val_loader, class_names, device, output_matrix_path="confusion_matrix.png"):
    """Performs detailed validation evaluation, outputs report, and saves confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluating model on final validation dataset...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n" + "="*60)
    print("FINAL VALIDATION CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60)
    
    # Save report to text file
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("Saved classification report to classification_report.txt")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix (DenseNet121)',
           ylabel='True Leaf Disease Label',
           xlabel='Predicted Leaf Disease Label')
           
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over grid cells and annotate with numbers
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                    
    fig.tight_layout()
    plt.savefig(output_matrix_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix visualization to {output_matrix_path}")


def predict_image(image_path, model_path, class_names_path, device):
    """Predicts the disease of a single plant leaf image."""
    # Load class names
    if not os.path.exists(class_names_path):
        print(f"Error: Class names mapping file '{class_names_path}' not found! Run training mode first to generate it.")
        return
        
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        print(f"Error loading class names: {e}")
        return
        
    num_classes = len(class_names)
    
    # Create model skeleton
    try:
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
    except AttributeError:
        model = models.densenet121(pretrained=True)
        
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, num_classes)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Model weights file '{model_path}' not found! Run training mode first to train the model.")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model weights from '{model_path}'")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model = model.to(device)
    model.eval()
    
    # Load and transform image
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return
        
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error reading image: {e}")
        return
        
    # Standard inference preprocessing
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Run prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_idx = torch.max(probabilities, 0)
        
    pred_class = class_names[pred_idx.item()]
    
    print("\n" + "="*60)
    print(f"DETECTION RESULTS FOR: {image_path}")
    print("="*60)
    print(f"Diagnosed Leaf State:  {pred_class}")
    print(f"Confidence score:      {confidence.item() * 100:.2f}%")
    print("="*60)
    
    # Print list of top predicted classes
    print("\nDetails (Class probabilities):")
    prob_dict = {class_names[i]: probabilities[i].item() * 100 for i in range(num_classes)}
    for name, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {name:<45} : {prob:.2f}%")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DenseNet121 Plant Disease Identification Pipeline")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                        help="Mode of operation: 'train' to train model, 'predict' to run inference on a single image.")
    parser.add_argument('--data-dir', type=str, default='.',
                        help="Directory path that contains the 'train' and 'valid' folders.")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of epochs to train.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="DataLoader batch size.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Optimizer learning rate.")
    parser.add_argument('--freeze-backbone', type=str2bool, default=True,
                        help="Whether to freeze DenseNet121 features. Highly recommended for fast CPU training.")
    parser.add_argument('--subset', type=float, default=1.0,
                        help="Fraction of dataset to use (from 0.01 to 1.0) to speed up training on CPU.")
    parser.add_argument('--model-path', type=str, default='densenet121_tomato.pth',
                        help="Path to save or load the model checkpoint.")
    parser.add_argument('--class-names-path', type=str, default='class_names.json',
                        help="Path to save or load the class names dictionary.")
    parser.add_argument('--image', type=str, default=None,
                        help="Path to the query leaf image for 'predict' mode.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
                        
    args = parser.parse_args()
    
    # Apply global seed
    set_seed(args.seed)
    
    # Auto-detect CUDA capability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")
    
    if args.mode == 'train':
        train_path = os.path.join(args.data_dir, 'train')
        valid_path = os.path.join(args.data_dir, 'valid')
        
        # Verify dataset paths
        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            print(f"Error: Could not find 'train/' or 'valid/' directories in '{args.data_dir}'!")
            sys.exit(1)
            
        print("Loading datasets...")
        full_train_dataset = ImageFolder(root=train_path, transform=train_transform)
        full_valid_dataset = ImageFolder(root=valid_path, transform=val_transform)
        
        class_names = full_train_dataset.classes
        num_classes = len(class_names)
        
        # Save class names mapping
        with open(args.class_names_path, 'w') as f:
            json.dump(class_names, f, indent=4)
        print(f"Saved class mapping schema to '{args.class_names_path}'")
        
        # Obtain subsets if specified
        train_dataset = get_subset(full_train_dataset, args.subset, args.seed)
        valid_dataset = get_subset(full_valid_dataset, args.subset, args.seed)
        
        print("\n--- Pipeline Parameters Summary ---")
        print(f" - Train directory:    {train_path}")
        print(f" - Validation directory: {valid_path}")
        print(f" - Number of Classes:  {num_classes}")
        print(f" - Epochs to train:    {args.epochs}")
        print(f" - Batch Size:          {args.batch_size}")
        print(f" - Learning Rate:      {args.lr}")
        print(f" - Freeze Backbone:    {args.freeze_backbone}")
        print(f" - Subset Fraction:    {args.subset}")
        print(f" - Total Train Images: {len(train_dataset)} (out of {len(full_train_dataset)})")
        print(f" - Total Valid Images: {len(valid_dataset)} (out of {len(full_valid_dataset)})")
        print("------------------------------------\n")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Construct the model
        print("Instantiating DenseNet121 model...")
        try:
            weights = models.DenseNet121_Weights.DEFAULT
            model = models.densenet121(weights=weights)
            print("Successfully loaded DenseNet121 pre-trained weights.")
        except AttributeError:
            model = models.densenet121(pretrained=True)
            print("Successfully loaded DenseNet121 pre-trained weights (legacy fallback).")
            
        # Freeze backbone parameters
        if args.freeze_backbone:
            print("Freezing DenseNet121 convolutional layers...")
            for param in model.parameters():
                param.requires_grad = False
                
        # Custom head classifier
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        
        # Ensure our custom head classifier weights are explicitly trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        model = model.to(device)
        
        # Setup Optimizer and Loss
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Train Model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.epochs,
            device=device,
            save_path=args.model_path
        )
        
        # Export metrics history to CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv("training_history.csv", index=False)
        print("Saved detailed metrics history to 'training_history.csv'")
        
        # Plot curves
        plot_curves(history)
        
        # Load best weights for final testing / validation analysis
        if os.path.exists(args.model_path):
            print("\nReloading best checkpoint weights for final evaluation...")
            model.load_state_dict(torch.load(args.model_path))
            
        evaluate_model(model, valid_loader, class_names, device)
        
    elif args.mode == 'predict':
        if not args.image:
            print("Error: You must specify a test image path using '--image <path>' in predict mode!")
            sys.exit(1)
        predict_image(args.image, args.model_path, args.class_names_path, device)


if __name__ == "__main__":
    main()
