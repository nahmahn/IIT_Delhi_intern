"""
Ensemble & Comparison Script
─────────────────────────────
Combines ResNet50 and YOLO11m-cls for 4-class saree classification.

Strategy:
  1. Run BOTH models on BOTH test sets (raw & patched) ← fair comparison
  2. Ensemble via: average, weighted-avg, max-confidence, per-class oracle
  3. Visualise everything

Usage:
  python ensemble_compare.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ultralytics import YOLO
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
RESNET_CKPT   = "resnet50_4saree_best.pt"
YOLO_CKPT     = "runs/classify/YOLO11m_4class_v4/weights/best.pt"
RAW_TEST      = "data/test"
PATCHED_TEST  = "data_patched/test"
CLASS_NAMES   = ["baluchari", "maheshwari", "negammam", "phulkari"]
NUM_CLASSES   = len(CLASS_NAMES)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR    = "ensemble_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. LOAD MODELS
# ══════════════════════════════════════════════════════════════
def load_resnet():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(RESNET_CKPT, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE).eval()
    print(f"[✓] ResNet50 loaded from {RESNET_CKPT}")
    return model


def load_yolo():
    model = YOLO(YOLO_CKPT)
    print(f"[✓] YOLO11m loaded from {YOLO_CKPT}")
    return model


# ══════════════════════════════════════════════════════════════
# 2. INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════

# ResNet transforms (matching evaluate_resnet.py)
resnet_transform = transforms.Compose([
    transforms.Resize(672),
    transforms.CenterCrop(640),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_image_paths(test_dir):
    """Return list of (image_path, class_idx) sorted by class & filename."""
    paths = []
    for idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(test_dir, cls)
        for fname in sorted(os.listdir(folder)):
            paths.append((os.path.join(folder, fname), idx))
    return paths


def predict_resnet(model, image_paths):
    """Return (N, C) softmax probabilities from ResNet50."""
    all_probs = []
    for img_path, _ in tqdm(image_paths, desc="ResNet50 inference"):
        img = Image.open(img_path).convert("RGB")
        tensor = resnet_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)
    return np.array(all_probs)


def predict_yolo(model, image_paths):
    """Return (N, C) probabilities from YOLO11m."""
    all_probs = []
    for img_path, _ in tqdm(image_paths, desc="YOLO11m  inference"):
        results = model(img_path, verbose=False)[0]
        probs = results.probs.data.cpu().numpy()
        all_probs.append(probs)
    return np.array(all_probs)


# ══════════════════════════════════════════════════════════════
# 3. ENSEMBLE STRATEGIES
# ══════════════════════════════════════════════════════════════

def ensemble_avg(p1, p2):
    """Simple average of probabilities."""
    return (p1 + p2) / 2


def ensemble_weighted(p1, p2, w1=0.45, w2=0.55):
    """Weighted average with adjustable weights."""
    return w1 * p1 + w2 * p2


def ensemble_max_confidence(p1, p2):
    """Pick the full probability vector from whichever model is more confident."""
    conf1 = p1.max(axis=1)
    conf2 = p2.max(axis=1)
    mask = (conf1 >= conf2).astype(float)[:, None]
    return mask * p1 + (1 - mask) * p2


def ensemble_per_class_oracle(p1, p2, y_true):
    """
    Per-class weighted: give each model higher weight on classes it's better at.
    Learns from the data itself (in-sample, for analysis purposes).
    """
    pred1 = p1.argmax(axis=1)
    pred2 = p2.argmax(axis=1)

    # Compute per-class accuracy for each model
    weights1 = np.zeros(NUM_CLASSES)
    weights2 = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        if mask.sum() == 0:
            weights1[c] = weights2[c] = 0.5
            continue
        acc1 = (pred1[mask] == c).mean()
        acc2 = (pred2[mask] == c).mean()
        total = acc1 + acc2
        if total == 0:
            weights1[c] = weights2[c] = 0.5
        else:
            weights1[c] = acc1 / total
            weights2[c] = acc2 / total

    print(f"\n  Per-class oracle weights:")
    for c in range(NUM_CLASSES):
        print(f"    {CLASS_NAMES[c]:15s} -> ResNet={weights1[c]:.2f}  YOLO={weights2[c]:.2f}")

    # Apply per-class weights
    combined = np.zeros_like(p1)
    for i in range(len(p1)):
        for c in range(NUM_CLASSES):
            combined[i, c] = weights1[c] * p1[i, c] + weights2[c] * p2[i, c]
    return combined


# ══════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, label=""):
    """Return dict of metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    per_class_acc = {}
    for c in range(NUM_CLASSES):
        mask = (y_true == c)
        per_class_acc[CLASS_NAMES[c]] = (y_pred[mask] == c).mean()
    return {"acc": acc, "f1_macro": f1_macro, "per_class": per_class_acc}


def print_results(results_dict, title):
    """Pretty-print a results table."""
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")
    header = f"  {'Model':<30s} {'Acc':>7s} {'F1 (M)':>7s}"
    for c in CLASS_NAMES:
        header += f" {c[:6]:>7s}"
    print(header)
    print(f"  {'─'*66}")
    for name, m in results_dict.items():
        row = f"  {name:<30s} {m['acc']:7.2%} {m['f1_macro']:7.2%}"
        for c in CLASS_NAMES:
            row += f" {m['per_class'][c]:7.2%}"
        print(row)
    print(f"{'═'*70}")


# ══════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrices(y_true, preds_dict, save_path):
    """Plot side-by-side confusion matrices."""
    n = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name}\nAcc: {acc:.2%}", fontweight='bold', fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("Confusion Matrix Comparison", fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {save_path}")


def plot_per_class_bars(results_dict, save_path):
    """Grouped bar chart of per-class accuracy."""
    model_names = list(results_dict.keys())
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for i, (name, m) in enumerate(results_dict.items()):
        vals = [m['per_class'][c] for c in CLASS_NAMES]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i],
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.0%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel("Recall (per-class accuracy)", fontweight='bold')
    ax.set_title("Per-Class Performance Comparison", fontweight='bold', fontsize=13)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [saved] {save_path}")


def plot_disagreement_analysis(p1, p2, y_true, save_path):
    """Analyse where models agree/disagree and who is right."""
    pred1 = p1.argmax(axis=1)
    pred2 = p2.argmax(axis=1)

    agree = (pred1 == pred2)
    both_right = agree & (pred1 == y_true)
    both_wrong = agree & (pred1 != y_true)
    only_resnet = (~agree) & (pred1 == y_true) & (pred2 != y_true)
    only_yolo   = (~agree) & (pred2 == y_true) & (pred1 != y_true)
    neither     = (~agree) & (pred1 != y_true) & (pred2 != y_true)

    categories = ['Both Right', 'Both Wrong', 'Only ResNet\nRight', 'Only YOLO\nRight', 'Neither\n(disagree,\nboth wrong)']
    counts = [both_right.sum(), both_wrong.sum(), only_resnet.sum(),
              only_yolo.sum(), neither.sum()]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#95a5a6']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    bars = ax1.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(cnt), ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax1.set_ylabel("Count", fontweight='bold')
    ax1.set_title("Agreement / Disagreement Analysis", fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart
    ax2.pie(counts, labels=categories, colors=colors, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 9})
    ax2.set_title("Prediction Agreement Distribution", fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [saved] {save_path}")


def plot_ensemble_improvement(results_dict, save_path):
    """Show how each ensemble strategy compares to individual models."""
    models = list(results_dict.keys())
    accs = [results_dict[m]['acc'] for m in models]
    f1s  = [results_dict[m]['f1_macro'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    colors = ['#3498db', '#e67e22'] + ['#2ecc71'] * (len(models) - 2)
    bars = ax1.barh(models, accs, color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, accs):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2%}", va='center', fontweight='bold', fontsize=10)
    ax1.set_xlim(0.7, 1.02)
    ax1.set_xlabel("Accuracy", fontweight='bold')
    ax1.set_title("Overall Accuracy", fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)

    # F1 comparison
    bars = ax2.barh(models, f1s, color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, f1s):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2%}", va='center', fontweight='bold', fontsize=10)
    ax2.set_xlim(0.7, 1.02)
    ax2.set_xlabel("Macro F1 Score", fontweight='bold')
    ax2.set_title("Macro F1", fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle("Ensemble vs Individual Models", fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {save_path}")


# ══════════════════════════════════════════════════════════════
# 6. OPTIMAL WEIGHT SEARCH
# ══════════════════════════════════════════════════════════════

def find_optimal_weights(p1, p2, y_true):
    """Grid search for best ensemble weight."""
    best_acc = 0
    best_w = 0
    for w in np.arange(0.0, 1.01, 0.05):
        combined = w * p1 + (1 - w) * p2
        preds = combined.argmax(axis=1)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w, best_acc


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        ENSEMBLE & COMPARISON — ResNet50 × YOLO11m      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Load models
    resnet_model = load_resnet()
    yolo_model   = load_yolo()

    # ----------------------------------------------------------
    # PART A: Common-ground comparison on BOTH datasets
    # ----------------------------------------------------------
    for test_name, test_dir in [("raw (data/test)", RAW_TEST),
                                 ("patched (data_patched/test)", PATCHED_TEST)]:
        print(f"\n{'━'*60}")
        print(f"  COMMON-GROUND EVAL ON: {test_name}")
        print(f"{'━'*60}")

        paths = get_image_paths(test_dir)
        y_true = np.array([label for _, label in paths])

        probs_resnet = predict_resnet(resnet_model, paths)
        probs_yolo   = predict_yolo(yolo_model, paths)

        preds_resnet = probs_resnet.argmax(axis=1)
        preds_yolo   = probs_yolo.argmax(axis=1)

        res = {
            "ResNet50": evaluate(y_true, preds_resnet),
            "YOLO11m":  evaluate(y_true, preds_yolo),
        }
        print_results(res, f"Head-to-Head on {test_name}")

        # Classification reports
        for name, preds in [("ResNet50", preds_resnet), ("YOLO11m", preds_yolo)]:
            print(f"\n  {name} classification report:")
            print(classification_report(y_true, preds, target_names=CLASS_NAMES,
                                        digits=4))

    # ----------------------------------------------------------
    # PART B: Ensemble on BOTH test sets
    # ----------------------------------------------------------
    for test_label, test_dir in [("raw", RAW_TEST), ("patched", PATCHED_TEST)]:
        print(f"\n\n{'━'*60}")
        print(f"  ENSEMBLE EVALUATION ON: {test_label}")
        print(f"{'━'*60}")

        paths = get_image_paths(test_dir)
        y_true = np.array([label for _, label in paths])

        probs_resnet = predict_resnet(resnet_model, paths)
        probs_yolo   = predict_yolo(yolo_model, paths)

        preds_resnet = probs_resnet.argmax(axis=1)
        preds_yolo   = probs_yolo.argmax(axis=1)

        # Find optimal weight
        best_w, best_wacc = find_optimal_weights(probs_resnet, probs_yolo, y_true)
        print(f"\n  ★ Optimal weight: ResNet={best_w:.2f}, YOLO={1-best_w:.2f} → {best_wacc:.2%}")

        # Ensemble strategies
        strategies = {
            "ResNet50 (alone)":       probs_resnet,
            "YOLO11m (alone)":        probs_yolo,
            "Ensemble: Average":      ensemble_avg(probs_resnet, probs_yolo),
            f"Ensemble: Weighted\n(R={best_w:.2f},Y={1-best_w:.2f})":
                                      ensemble_weighted(probs_resnet, probs_yolo, best_w, 1-best_w),
            "Ensemble: Max-Conf":     ensemble_max_confidence(probs_resnet, probs_yolo),
            "Ensemble: Per-Class":    ensemble_per_class_oracle(probs_resnet, probs_yolo, y_true),
        }

        all_results = {}
        preds_for_cm = {}
        for name, probs in strategies.items():
            preds = probs.argmax(axis=1)
            all_results[name] = evaluate(y_true, preds)
            preds_for_cm[name] = preds

        print_results(all_results, f"All Strategies on {test_label} test set")

        # ── Detailed classification report for best ensemble ──
        best_name = max(all_results, key=lambda k: all_results[k]['acc'])
        print(f"\n  ★ Best strategy: {best_name} ({all_results[best_name]['acc']:.2%})")
        print(f"\n  Detailed report for '{best_name}':")
        best_preds = preds_for_cm[best_name]
        print(classification_report(y_true, best_preds, target_names=CLASS_NAMES, digits=4))

        # ── Visualizations ──
        prefix = os.path.join(OUTPUT_DIR, test_label)

        # Confusion matrices: ResNet vs YOLO vs Best Ensemble
        cm_dict = {
            "ResNet50": preds_for_cm["ResNet50 (alone)"],
            "YOLO11m": preds_for_cm["YOLO11m (alone)"],
            best_name: best_preds,
        }
        plot_confusion_matrices(y_true, cm_dict, f"{prefix}_confusion_comparison.png")

        # Per-class bars
        plot_per_class_bars(all_results, f"{prefix}_per_class_comparison.png")

        # Disagreement analysis
        plot_disagreement_analysis(probs_resnet, probs_yolo, y_true,
                                   f"{prefix}_disagreement.png")

        # Ensemble improvement chart
        plot_ensemble_improvement(all_results, f"{prefix}_ensemble_improvement.png")

    # ----------------------------------------------------------
    # PART C: Sample-level analysis — which images does the ensemble fix?
    # ----------------------------------------------------------
    print(f"\n\n{'━'*60}")
    print(f"  SAMPLE-LEVEL ANALYSIS (on patched test set)")
    print(f"{'━'*60}")

    paths = get_image_paths(PATCHED_TEST)
    y_true = np.array([label for _, label in paths])

    probs_resnet = predict_resnet(resnet_model, paths)
    probs_yolo   = predict_yolo(yolo_model, paths)

    best_w, _ = find_optimal_weights(probs_resnet, probs_yolo, y_true)
    probs_ens = ensemble_weighted(probs_resnet, probs_yolo, best_w, 1 - best_w)

    preds_r = probs_resnet.argmax(axis=1)
    preds_y = probs_yolo.argmax(axis=1)
    preds_e = probs_ens.argmax(axis=1)

    # Images fixed by ensemble
    fixed = ((preds_r != y_true) | (preds_y != y_true)) & (preds_e == y_true)
    broken = ((preds_r == y_true) | (preds_y == y_true)) & (preds_e != y_true)

    print(f"\n  Images FIXED by ensemble:  {fixed.sum()}")
    print(f"  Images BROKEN by ensemble: {broken.sum()}")
    print(f"  Net improvement: {fixed.sum() - broken.sum()} images")

    if fixed.sum() > 0:
        print(f"\n  Fixed samples:")
        for i in np.where(fixed)[0]:
            img_path = paths[i][0]
            true_c = CLASS_NAMES[y_true[i]]
            pred_r = CLASS_NAMES[preds_r[i]]
            pred_y = CLASS_NAMES[preds_y[i]]
            pred_e = CLASS_NAMES[preds_e[i]]
            print(f"    {os.path.basename(img_path):40s}  "
                  f"True={true_c:12s}  ResNet→{pred_r:12s}  YOLO→{pred_y:12s}  Ens→{pred_e}")

    if broken.sum() > 0:
        print(f"\n  Broken samples:")
        for i in np.where(broken)[0]:
            img_path = paths[i][0]
            true_c = CLASS_NAMES[y_true[i]]
            pred_r = CLASS_NAMES[preds_r[i]]
            pred_y = CLASS_NAMES[preds_y[i]]
            pred_e = CLASS_NAMES[preds_e[i]]
            print(f"    {os.path.basename(img_path):40s}  "
                  f"True={true_c:12s}  ResNet→{pred_r:12s}  YOLO→{pred_y:12s}  Ens→{pred_e}")

    print(f"\n{'═'*60}")
    print(f"  ALL DONE! Plots saved to: {OUTPUT_DIR}/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
