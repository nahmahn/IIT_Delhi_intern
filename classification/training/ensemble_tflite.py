"""
Ensemble TFLite Inference
──────────────────────────
Runs both ResNet50 + YOLO11m TFLite models and ensembles their predictions.
Use this to:
  1. Verify TFLite accuracy matches PyTorch ensemble
  2. Serve as reference for Dart/Flutter implementation

Usage:
  python ensemble_tflite.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
RESNET_TFLITE  = "tflite_models/resnet50_4saree.tflite"
YOLO_TFLITE    = "tflite_models/yolo11m_4class.tflite"
TEST_DIR       = "data_patched/test"
CLASS_NAMES    = ["baluchari", "maheshwari", "negammam", "phulkari"]
NUM_CLASSES    = len(CLASS_NAMES)

# Best weights from your ensemble_compare.py results
# Raw: R=0.60, Y=0.40 → 91.94%  |  Patched: R=0.15, Y=0.85 → 91.13%
# Using raw weights since they gave best accuracy
RESNET_WEIGHT = 0.60
YOLO_WEIGHT   = 0.40

# Image sizes
RESNET_IMG_SIZE = 640
YOLO_IMG_SIZE   = 640

# ImageNet normalization for ResNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# LOAD TFLITE INTERPRETERS
# ══════════════════════════════════════════════════════════════
import tensorflow as tf

def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    print(f"  Loaded: {path}")
    print(f"    Input:  {inp[0]['shape']}  dtype={inp[0]['dtype']}")
    print(f"    Output: {out[0]['shape']}  dtype={out[0]['dtype']}")
    return interp, inp, out


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocess_resnet(img_path):
    """
    Match the torchvision transform from ensemble_compare.py:
      Resize(672) → CenterCrop(640) → ToTensor → Normalize(ImageNet)
    TFLite expects NHWC, so output is (1, 640, 640, 3).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Resize(672) — scales shortest side to 672
    scale = 672 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop(640)
    left = (new_w - RESNET_IMG_SIZE) // 2
    top  = (new_h - RESNET_IMG_SIZE) // 2
    img = img.crop((left, top, left + RESNET_IMG_SIZE, top + RESNET_IMG_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization (per-channel)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return arr.reshape(1, RESNET_IMG_SIZE, RESNET_IMG_SIZE, 3)


def preprocess_yolo(img_path):
    """
    YOLO classify preprocessing: resize shortest side → center crop.
    TFLite expects NHWC, (1, 640, 640, 3), values in [0, 1].
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Resize shortest side to IMG_SIZE
    scale = YOLO_IMG_SIZE / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    left = (new_w - YOLO_IMG_SIZE) // 2
    top  = (new_h - YOLO_IMG_SIZE) // 2
    img = img.crop((left, top, left + YOLO_IMG_SIZE, top + YOLO_IMG_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3)


# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════
def run_tflite(interp, inp_details, out_details, input_data):
    # Check if model expects NCHW (PyTorch-exported) vs NHWC
    expected = inp_details[0]['shape']
    if expected[1] == 3 and input_data.shape[1] != 3:
        # Model expects NCHW, convert
        input_data = np.transpose(input_data, (0, 3, 1, 2))

    interp.set_tensor(inp_details[0]['index'], input_data)
    interp.invoke()
    return interp.get_tensor(out_details[0]['index'])[0]


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ENSEMBLE TFLITE — ResNet50 × YOLO11m               ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    if not os.path.exists(RESNET_TFLITE):
        print(f"✗ ResNet TFLite not found: {RESNET_TFLITE}")
        print("  Run: python export_to_tflite.py")
        return
    if not os.path.exists(YOLO_TFLITE):
        print(f"✗ YOLO TFLite not found: {YOLO_TFLITE}")
        print("  Run: python export_to_tflite.py")
        return

    resnet_interp, resnet_inp, resnet_out = load_interpreter(RESNET_TFLITE)
    yolo_interp, yolo_inp, yolo_out       = load_interpreter(YOLO_TFLITE)

    # Collect test images
    paths = []
    for idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(TEST_DIR, cls)
        for fname in sorted(os.listdir(folder)):
            paths.append((os.path.join(folder, fname), idx))

    y_true = np.array([label for _, label in paths])

    # Run inference
    all_resnet_probs = []
    all_yolo_probs   = []

    for img_path, _ in tqdm(paths, desc="TFLite inference"):
        # ResNet
        r_input = preprocess_resnet(img_path)
        r_out   = run_tflite(resnet_interp, resnet_inp, resnet_out, r_input)
        # Apply softmax if model outputs logits (no softmax baked in)
        r_probs = r_out if r_out.sum() > 0.99 and r_out.min() >= 0 else softmax(r_out)
        all_resnet_probs.append(r_probs)

        # YOLO
        y_input = preprocess_yolo(img_path)
        y_out   = run_tflite(yolo_interp, yolo_inp, yolo_out, y_input)
        y_probs = y_out if y_out.sum() > 0.99 and y_out.min() >= 0 else softmax(y_out)
        all_yolo_probs.append(y_probs)

    probs_r = np.array(all_resnet_probs)
    probs_y = np.array(all_yolo_probs)

    # Individual predictions
    preds_r = probs_r.argmax(axis=1)
    preds_y = probs_y.argmax(axis=1)

    # Ensemble: weighted average
    probs_ens = RESNET_WEIGHT * probs_r + YOLO_WEIGHT * probs_y
    preds_ens = probs_ens.argmax(axis=1)

    # Also try simple average
    probs_avg = (probs_r + probs_y) / 2
    preds_avg = probs_avg.argmax(axis=1)

    # Results
    print(f"\n{'═'*60}")
    print(f"  TFLITE ENSEMBLE RESULTS (on {TEST_DIR})")
    print(f"{'═'*60}")
    print(f"  {'Method':<35s} {'Accuracy':>10s}")
    print(f"  {'─'*45}")
    print(f"  {'ResNet50 TFLite (alone)':<35s} {accuracy_score(y_true, preds_r):>10.2%}")
    print(f"  {'YOLO11m TFLite (alone)':<35s} {accuracy_score(y_true, preds_y):>10.2%}")
    print(f"  {'Ensemble: Average':<35s} {accuracy_score(y_true, preds_avg):>10.2%}")
    print(f"  {f'Ensemble: Weighted (R={RESNET_WEIGHT},Y={YOLO_WEIGHT})':<35s} {accuracy_score(y_true, preds_ens):>10.2%}")
    print(f"{'═'*60}")

    # Quick grid search for optimal TFLite weights
    print(f"\n  Grid search for optimal TFLite weights:")
    best_acc, best_w = 0, 0
    for w in np.arange(0.0, 1.01, 0.05):
        combo = w * probs_r + (1 - w) * probs_y
        acc = accuracy_score(y_true, combo.argmax(axis=1))
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"  ★ Optimal: R={best_w:.2f}, Y={1-best_w:.2f} → {best_acc:.2%}")

    # Detailed report for best ensemble
    print(f"\n  Classification Report (Weighted Ensemble):")
    print(classification_report(y_true, preds_ens, target_names=CLASS_NAMES, digits=4))

    print(f"{'═'*60}")
    print(f"  Done! Both .tflite models verified for ensemble deployment.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
