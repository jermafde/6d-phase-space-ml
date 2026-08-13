"""
ResNet18-based CNN for muon beam momentum prediction.
Reads 2D histogram PNG images (x position vs detector z location)
and predicts the initial momentum magnitude p0 (MeV/c).

ResNet18 replaces the custom CNN from the previous version.
All data loading, augmentation, folder structure, TRAIN_MODE flag,
and histogram plots remain identical to the previous version.

Key changes from custom CNN version:
  - Model: ResNet18 (pretrained=False, modified for grayscale + regression)
  - Input channels changed from 3 to 1 (grayscale)
  - Final classification layer replaced with single regression output
  - Loss: MSELoss (same as before)
  - IMG_SIZE increased to 224 to match ResNet18's expected input size

=====================================================================
FOLDER STRUCTURE REQUIRED
=====================================================================
DATA_DIR/
    130Histo/
        sample_0.png
        sample_1.png
        ... (100 files, one per random 100-event sample)
    140Histo/
        sample_0.png
        ...
    200Histo/
        sample_0.png
        ...

The p0 label is extracted from the subfolder name.
Folder name must contain the momentum value as a number.

=====================================================================
TRAIN_MODE GUIDE
=====================================================================
TRAIN_MODE = True
  - Loads all PNG images from DATA_DIR subfolders
  - Applies light data augmentation
  - Trains ResNet18 with 80/20 train/test split
  - Saves best model to MODEL_SAVE_PATH
  - Plots training/val loss and actual vs predicted histograms

TRAIN_MODE = False
  - Loads saved model from MODEL_SAVE_PATH
  - Evaluates on PNG images in EVAL_DIR
  - Computes RMSE, MAE, mean prediction per p0
  - Plots actual vs predicted histograms
=====================================================================
"""

# Imports
import os
import re
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# *** CHANGE THESE TO CONTROL THE SCRIPT ***
# ==========================================

# True  — train and save model
# False — load saved model and evaluate
TRAIN_MODE = True

# Directory containing momentum subfolders with PNG files
DATA_DIR = "/Users/jayshirlee/Library/CloudStorage/GoogleDrive-jayeushirlee06@gmail.com/My Drive/AI"

# Directory to evaluate on when TRAIN_MODE = False
EVAL_DIR = "/Users/jayshirlee/Library/CloudStorage/GoogleDrive-jayeushirlee06@gmail.com/My Drive/AI"

# Where to save / load the trained model weights
MODEL_SAVE_PATH = "/Users/jayshirlee/Desktop/muon_resnet_pngv18.pth"

# p0 values to show in evaluation histogram plots
# Auto-detected from folder names in True mode
# Override manually for False mode if needed
P0_VALUES = [127.5, 132.5, 137.5, 142.5, 147.5, 152.5, 157.5, 162.5, 167.5, 172.5, 177.5, 182.5]
# p0 normalization range — set to cover full expected momentum range
P0_MIN = 125.0
P0_MAX = 185.0

# Image size — ResNet18 works best at 224x224
IMG_SIZE = 224

# Number of augmented copies per original image
# With 100 PNGs per momentum value you have much more data than before
# Keep augmentation light — 5 to 10 is enough
NUM_AUGMENTATIONS = 5

# ==========================================
# HYPERPARAMETERS
# ==========================================
learning_rate = 1e-4    # Lower LR for ResNet — it's more sensitive than custom CNN
batch_size    = 16
num_epochs    = 50      # ResNet converges faster than custom CNN


# ==========================================
# PART 1: IMAGE LOADING AND AUGMENTATION
# ==========================================
def load_single_png(image_path):
    """
    Loads one histogram PNG as a grayscale numpy array.
    Resizes to IMG_SIZE x IMG_SIZE and normalizes to [0, 1].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_resized    = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype("float32") / 255.0
    return img_normalized


def augment_image(img):
    """
    Light augmentation — keeps histogram shape intact.
    Only brightness, contrast, and small noise.
    No pixel shifts since the x/z axis positions are meaningful.
    """
    aug = img.copy()

    # Random brightness
    brightness = np.random.uniform(-0.05, 0.05)
    aug = np.clip(aug + brightness, 0.0, 1.0)

    # Random contrast
    contrast = np.random.uniform(0.95, 1.05)
    aug = np.clip(aug * contrast, 0.0, 1.0)

    # Small Gaussian noise
    noise = np.random.normal(0, 0.008, aug.shape).astype("float32")
    aug = np.clip(aug + noise, 0.0, 1.0)

    return aug


def extract_p0_from_folder(folder_name):
    """Extracts p0 value from folder name. '130Histo' -> 130.0"""
    match = re.search(r'(\d+\.?\d*)', folder_name)
    return float(match.group(1)) if match else None


def load_all_images(base_dir, augment=False):
    """
    Walks through base_dir subfolders, loads all PNG files,
    labels each with the p0 extracted from the subfolder name.
    Optionally applies augmentation.

    Returns:
        images : np.array shape (N, IMG_SIZE, IMG_SIZE, 1)
        labels : np.array shape (N,) in raw MeV/c
    """
    images = []
    labels = []

    subfolders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    if not subfolders:
        raise ValueError(f"No subfolders found in {base_dir}")

    print(f"\nFound {len(subfolders)} momentum folders:")

    for folder in subfolders:
        p0 = extract_p0_from_folder(folder)
        if p0 is None:
            print(f"  Skipping '{folder}' — no number found in name")
            continue

        folder_path = os.path.join(base_dir, folder)
        png_files   = sorted(glob.glob(os.path.join(folder_path, "*.png")))

        if not png_files:
            print(f"  Skipping '{folder}' — no PNG files found")
            continue

        print(f"  {folder}  ->  p0={p0:.1f} MeV/c  |  {len(png_files)} PNGs")

        for png_path in png_files:
            try:
                img = load_single_png(png_path)

                # Add original image
                images.append(img[..., np.newaxis])
                labels.append(p0)

                # Add augmented copies
                if augment:
                    for _ in range(NUM_AUGMENTATIONS):
                        aug_img = augment_image(img)
                        images.append(aug_img[..., np.newaxis])
                        labels.append(p0)

            except Exception as e:
                print(f"    Error loading {png_path}: {e}")

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="float32")

    print(f"\nTotal samples loaded: {len(labels)}")
    print(f"p0 values found: {np.unique(labels)}")
    print(f"Image array shape: {images.shape}")

    return images, labels


# ==========================================
# PART 2: PYTORCH DATASET
# ==========================================
class PNGHistogramDataset(Dataset):
    """
    PyTorch Dataset wrapping numpy image arrays and p0 labels.
    Normalizes p0 to [0, 1] for training.
    Stores raw p0 for RMSE reporting in MeV/c.
    """
    def __init__(self, images, labels, p0_min=P0_MIN, p0_max=P0_MAX):
        self.p0_min = p0_min
        self.p0_max = p0_max

        # Images: (N, H, W, 1) -> (N, 1, H, W) for PyTorch conv layers
        imgs_transposed  = np.transpose(images, (0, 3, 1, 2))
        self.images  = torch.tensor(imgs_transposed, dtype=torch.float32)

        # Normalize p0 to [0, 1]
        p0_norm      = (labels - p0_min) / (p0_max - p0_min)
        self.p0      = torch.tensor(p0_norm,  dtype=torch.float32)
        self.p0_raw  = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.p0)

    def __getitem__(self, idx):
        return self.images[idx], self.p0[idx].unsqueeze(0), self.p0_raw[idx].unsqueeze(0)


# ==========================================
# PART 3: RESNET18 MODEL
# ==========================================
def build_resnet18():
    """
    Builds a ResNet18 modified for:
      - Grayscale input (1 channel instead of 3)
      - Regression output (1 continuous value instead of 1000 classes)
      - No pretrained weights (training from scratch on physics data)

    The first conv layer is replaced to accept 1-channel input.
    The final fully connected layer is replaced for regression.
    """
    # Load ResNet18 architecture without pretrained weights
    model = models.resnet18(weights=None)

    # Replace first conv layer: 3 channels -> 1 channel
    # Keep all other parameters the same
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Replace final classification layer with regression output
    # model.fc.in_features is 512 for ResNet18
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model


# ==========================================
# PART 4: DEVICE AND MODEL INIT
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Mode: {'TRAINING' if TRAIN_MODE else 'EVALUATION'}")

model     = build_resnet18().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

epoch_train_losses = []
epoch_val_losses   = []

# Print model summary
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ResNet18 trainable parameters: {total_params:,}")


# ==========================================
# PART 5: TRAIN OR LOAD
# ==========================================
if TRAIN_MODE == True:
    # -----------------------------------------
    # TRAINING MODE
    # -----------------------------------------
    print(f"\nLoading training images from: {DATA_DIR}")
    images, labels = load_all_images(DATA_DIR, augment=True)

    # Update P0_VALUES from actual data
    P0_VALUES = sorted(np.unique(labels).tolist())

    dataset = PNGHistogramDataset(images, labels)

    # Fixed seed for reproducible split
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test  samples: {len(test_dataset)}")

    # Scheduler reduces LR if val loss stalls for 8 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    print(f"\nStarting training ({num_epochs} epochs)...")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        # --- Training pass ---
        model.train()
        running_train_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=False)

        for img_batch, p0_norm_batch, _ in loop:
            img_batch     = img_batch.to(device)
            p0_norm_batch = p0_norm_batch.to(device)

            predictions = model(img_batch)
            loss = criterion(predictions, p0_norm_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * img_batch.size(0)
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_losses.append(epoch_train_loss)

        # --- Validation pass ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for img_batch, p0_norm_batch, _ in test_loader:
                img_batch     = img_batch.to(device)
                p0_norm_batch = p0_norm_batch.to(device)
                preds         = model(img_batch)
                running_val_loss += criterion(preds, p0_norm_batch).item() * img_batch.size(0)

        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        epoch_val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print(f"Epoch {epoch+1:3d}/{num_epochs}  "
              f"Train: {epoch_train_loss:.6f}  "
              f"Val: {epoch_val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"\nBest model saved to: {MODEL_SAVE_PATH}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Load best checkpoint for evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

elif TRAIN_MODE == False:
    # -----------------------------------------
    # EVALUATION MODE
    # -----------------------------------------
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No saved model found at {MODEL_SAVE_PATH}\n"
            f"Run with TRAIN_MODE = True first."
        )
    print(f"\nLoading saved model from: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("Model loaded successfully")

    print(f"\nLoading evaluation images from: {EVAL_DIR}")
    images, labels = load_all_images(EVAL_DIR, augment=False)

    eval_dataset = PNGHistogramDataset(images, labels)
    test_loader  = DataLoader(eval_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

else:
    raise ValueError(f"TRAIN_MODE must be True or False. Got: {TRAIN_MODE}")


# ==========================================
# PART 6: EVALUATION
# ==========================================
print("\nEvaluating...")
model.eval()

all_predicted = []
all_actual    = []

with torch.no_grad():
    for img_batch, p0_norm_batch, p0_raw_batch in test_loader:
        img_batch = img_batch.to(device)

        # Predict normalized p0 then denormalize to MeV/c
        preds_norm = model(img_batch).cpu().numpy()
        preds_raw  = preds_norm * (P0_MAX - P0_MIN) + P0_MIN

        all_predicted.append(preds_raw)
        all_actual.append(p0_raw_batch.numpy())

all_predicted = np.concatenate(all_predicted).flatten()
all_actual    = np.concatenate(all_actual).flatten()

rmse = np.sqrt(np.mean((all_predicted - all_actual) ** 2))
mae  = np.mean(np.abs(all_predicted - all_actual))

print(f"\nTest RMSE: {rmse:.3f} MeV/c")
print(f"Test MAE:  {mae:.3f} MeV/c")

for p0_val in P0_VALUES:
    mask = all_actual == p0_val
    if mask.sum() > 0:
        mean_pred = np.mean(all_predicted[mask])
        print(f"  p0={p0_val:.1f} MeV/c  ->  "
            f"Mean Predicted: {mean_pred:.3f} MeV/c  |  "
            f"Error: {mean_pred - p0_val:+.3f} MeV/c")


# ==========================================
# PART 7: PLOTS
# ==========================================

# --- Plot 1: Training and validation loss ---
if TRAIN_MODE == True and epoch_train_losses:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_train_losses,
             color="green", linewidth=2, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), epoch_val_losses,
             color="red", linewidth=2, linestyle="--", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (normalized p0)")
    plt.title("ResNet18 Training and Validation Loss Over Epochs")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/Users/jayshirlee/Desktop/resnet_training_loss.png", dpi=150)
    plt.show()

# --- Plot 2: Overlaid histogram — actual vs predicted ---
colors_actual = ["steelblue", "darkorange", "green",
                 "purple",    "brown",      "pink",
                 "cyan",      "magenta",    "yellow", "gray"]
colors_pred   = ["blue",      "red",        "darkgreen",
                 "darkviolet","saddlebrown","deeppink",
                 "darkcyan",  "darkmagenta","olive",  "black"]

fig, axes = plt.subplots(1, len(P0_VALUES),
                          figsize=(5 * len(P0_VALUES), 5), sharey=True)

if len(P0_VALUES) == 1:
    axes = [axes]

for idx, p0_val in enumerate(P0_VALUES):
    mask = all_actual == p0_val
    actual_subset    = all_actual[mask]
    predicted_subset = all_predicted[mask]

    if len(actual_subset) == 0:
        continue

    all_vals = np.concatenate([actual_subset, predicted_subset])
    bins = np.linspace(all_vals.min() - 5, all_vals.max() + 5, 60)

    axes[idx].hist(actual_subset,    bins=bins, alpha=0.6,
                   color=colors_actual[idx % len(colors_actual)],
                   label=f"Actual p0={p0_val}")
    axes[idx].hist(predicted_subset, bins=bins, alpha=0.6,
                   color=colors_pred[idx % len(colors_pred)],
                   label="Predicted")

    axes[idx].set_xlabel("Momentum Magnitude (MeV/c)", fontsize=12)
    axes[idx].set_ylabel("Events" if idx == 0 else "", fontsize=12)
    axes[idx].set_title(f"p0 = {p0_val} MeV/c",
                         fontsize=13, fontweight="bold")
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

mode_label = "Train+Test" if TRAIN_MODE == True else "Evaluation"
fig.suptitle(
    f"Actual vs Predicted Momentum  |  "
    f"RMSE: {rmse:.3f} MeV/c  MAE: {mae:.3f} MeV/c  |  {mode_label}",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/Users/jayshirlee/Desktop/resnet_momentum_prediction.png", dpi=150)
plt.show()

print("\nPlots saved to Desktop.")
