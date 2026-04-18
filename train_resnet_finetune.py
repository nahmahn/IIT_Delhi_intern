"""
IndoFashion ResNet50 → 4-Saree Fine-tuning Script
==================================================
3-stage transfer: ImageNet → IndoFashion (15 Indian clothes) → 4 Sarees

Checkpoint: iew_r50_jitter_flip.pt (IndoFashion ResNet50, 15-class head)
    Key structure: top-level keys = IndoFashion-trained weights
                   'resnet_model.*' keys = original ImageNet weights (ignored)

We use the TOP-LEVEL keys (IndoFashion-trained), replace fc(2048→15) with fc(2048→4),
then fine-tune in two phases:
    Phase 1: Head only (frozen backbone)
    Phase 2: Full model (gradual unfreeze, lower LR)
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.cuda.amp import GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
DATA_DIR          = "data_patched"
CHECKPOINT_PATH   = "iew_r50_jitter_flip.pt"           # IndoFashion R50
OUTPUT_BEST       = "resnet50_4saree_best.pt"           # Best checkpoint
OUTPUT_FINAL      = "resnet50_4saree_final.pt"          # Final checkpoint

BATCH_SIZE        = 8  # Reduced to avoid OOM at massive 640x640 resolution
WORKERS           = 0  # Windows requirement (set to 4 on Linux)

# Phase 1: head only
EPOCHS_P1         = 10
LR_P1             = 1e-3

# Phase 2: full fine-tune
EPOCHS_P2         = 50
LR_P2             = 3e-5     # 30× lower than head LR

PATIENCE          = 10       # Early stopping patience (per phase)
NUM_CLASSES       = 4        # Baluchari, Maheshwari, Negammam, Phulkari
LABEL_SMOOTHING   = 0.1      # Match your YOLO setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data Preparation
# ============================================================
def build_dataloaders():
    """Build train/val/test dataloaders with ResNet-style preprocessing."""
    xforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(640, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ]),
        "val": transforms.Compose([
            transforms.Resize(672),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }
    # Test uses same transforms as val
    xforms["test"] = xforms["val"]

    ds = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, split)
        if os.path.isdir(path):
            ds[split] = datasets.ImageFolder(path, xforms.get(split, xforms["val"]))

    loaders = {
        s: DataLoader(d, batch_size=BATCH_SIZE,
                      shuffle=(s == "train"), num_workers=WORKERS,
                      pin_memory=True)
        for s, d in ds.items()
    }

    for s, d in ds.items():
        print(f"  {s}: {len(d)} images, classes={d.classes}")

    return loaders, ds


# ============================================================
# Model: Load IndoFashion weights correctly
# ============================================================
def build_model():
    """
    Load ResNet50 with IndoFashion-trained weights.

    The checkpoint has two namespaces:
      - 'resnet_model.*' = original ImageNet weights (1000-class fc) — SKIP
      - top-level keys   = IndoFashion fine-tuned weights (15-class fc) — USE THESE

    We extract top-level keys, load into a standard ResNet50,
    then replace the 15-class head with a 4-class head.
    """
    print(f"\n{'='*60}")
    print(f"Loading IndoFashion ResNet50 from: {CHECKPOINT_PATH}")
    print(f"{'='*60}")

    # 1. Create standard ResNet50 with 15-class head (to match checkpoint)
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features  # 2048
    model.fc = nn.Linear(num_ftrs, 15)

    # 2. Load checkpoint and extract the top-level (IndoFashion-trained) keys
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    # Filter: keep only top-level keys (no 'resnet_model.' prefix)
    indofashion_weights = {
        k: v for k, v in ckpt.items()
        if not k.startswith("resnet_model.")
    }

    # Load into model
    missing, unexpected = model.load_state_dict(indofashion_weights, strict=False)
    loaded = len(indofashion_weights) - len(unexpected)
    print(f"  [OK] Loaded {loaded} IndoFashion-trained parameters")
    if missing:
        print(f"  [!] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [!] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    # 3. Verify the fc layer loaded the 15-class head correctly
    assert model.fc.weight.shape == (15, 2048), \
        f"fc shape mismatch: {model.fc.weight.shape}"
    print(f"  [OK] 15-class IndoFashion head verified: {model.fc.weight.shape}")

    # 4. Replace head with our 4-class head
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    nn.init.kaiming_normal_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    print(f"  [OK] Replaced head: fc(2048 -> {NUM_CLASSES})")

    return model.to(device)


# ============================================================
# Training Loop
# ============================================================
def train_phase(model, loaders, ds_sizes, optimizer, scheduler,
                num_epochs, phase_name, scaler):
    """Train one phase with early stopping, AMP, and LR scheduling."""
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n[{phase_name}] Epoch {epoch}/{num_epochs}")

        for split in ["train", "val"]:
            is_train = (split == "train")
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_correct = 0

            # Enhanced tqdm with live metric tracking
            pbar = tqdm(loaders[split], desc=f"{split:5s}", leave=False)
            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    with amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = outputs.argmax(1)

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                current_batch_loss = loss.item()
                current_batch_acc = (preds == labels).sum().item() / inputs.size(0)
                
                running_loss += current_batch_loss * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                
                # Update progress bar with live metrics
                pbar.set_postfix(loss=f"{current_batch_loss:.4f}", acc=f"{current_batch_acc:.4f}")

            epoch_loss = running_loss / ds_sizes[split]
            epoch_acc = running_correct / ds_sizes[split]
            print(f"  {split.upper():5s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if split == "val":
                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_wts, OUTPUT_BEST)
                    counter = 0
                    print(f"  * New Best Val Acc: {best_acc:.4f} (Saved to {OUTPUT_BEST})")
                else:
                    counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR: {current_lr:.2e}  |  Patience: {counter}/{PATIENCE}")

        if counter >= PATIENCE:
            print(f"  [!] Early stopping at epoch {epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_wts)
    print(f"\n  Best {phase_name} Accuracy: {best_acc:.4f}")
    return model, best_acc


# ============================================================
# Evaluation
# ============================================================
def evaluate_test(model, loaders, ds_sizes):
    """Evaluate on test set with per-class metrics."""
    if "test" not in loaders:
        print("No test split found, skipping.")
        return

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loaders["test"], desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {correct}/{total} = {correct/total:.4f}")
    print(f"{'='*60}")

    # Per-class accuracy
    class_names = loaders["test"].dataset.classes
    for i, name in enumerate(class_names):
        cls_total = sum(1 for l in all_labels if l == i)
        cls_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == i and p == i)
        acc = cls_correct / cls_total if cls_total > 0 else 0
        print(f"  {name:20s}: {cls_correct}/{cls_total} = {acc:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    loaders, ds_dict = build_dataloaders()
    ds_sizes = {s: len(d) for s, d in ds_dict.items()}

    # Model
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  |  Trainable: {trainable:,}")

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Phase 1: Head only ──────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: Train classification head only (backbone frozen)")
    print(f"{'='*60}")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    opt1 = optim.AdamW(model.fc.parameters(), lr=LR_P1, weight_decay=0.01)
    sched1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode="min",
                                                   factor=0.5, patience=3)
    model, acc1 = train_phase(model, loaders, ds_sizes, opt1, sched1,
                              EPOCHS_P1, "Phase 1 (Head)", scaler)

    # ── Phase 2: Full fine-tune ─────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tune entire model")
    print(f"{'='*60}")
    for p in model.parameters():
        p.requires_grad = True

    # Differential LR: backbone gets 10× lower LR than head
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = list(model.fc.parameters())
    opt2 = optim.AdamW([
        {"params": backbone_params, "lr": LR_P2},
        {"params": head_params,     "lr": LR_P2 * 10},
    ], weight_decay=5e-4)
    sched2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min",
                                                   factor=0.5, patience=4)
    model, acc2 = train_phase(model, loaders, ds_sizes, opt2, sched2,
                              EPOCHS_P2, "Phase 2 (Full)", scaler)

    # Save final
    torch.save(model.state_dict(), OUTPUT_FINAL)

    # ── Test ────────────────────────────────────────────────
    evaluate_test(model, loaders, ds_sizes)

    print(f"\n[DONE] Training complete!")
    print(f"   Phase 1 best val acc: {acc1:.4f}")
    print(f"   Phase 2 best val acc: {acc2:.4f}")
    print(f"   Best model: {OUTPUT_BEST}")
    print(f"   Final model: {OUTPUT_FINAL}")


if __name__ == "__main__":
    freeze_support()
    main()
