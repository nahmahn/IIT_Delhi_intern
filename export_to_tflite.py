"""
Export both models to TFLite
─────────────────────────────
1. YOLO11m  → ultralytics built-in export
2. ResNet50 → PyTorch → ONNX → TFLite (via ai-edge-torch)

Prerequisites:
  pip install ultralytics ai-edge-torch onnx

Usage:
  python export_to_tflite.py
"""

import os, sys, torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
RESNET_CKPT   = "resnet50_4saree_best.pt"
YOLO_CKPT     = "runs/classify/YOLO11m_4class_v4/weights/best.pt"
CLASS_NAMES   = ["baluchari", "maheshwari", "negammam", "phulkari"]
NUM_CLASSES   = len(CLASS_NAMES)

RESNET_IMG_SIZE = 640   # matches your training / ensemble_compare transform
YOLO_IMG_SIZE   = 640   # YOLO classify default

OUTPUT_DIR = "tflite_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. EXPORT YOLO11m → TFLite
# ══════════════════════════════════════════════════════════════
def export_yolo():
    from ultralytics import YOLO
    print("\n[1/2] Exporting YOLO11m to TFLite...")
    model = YOLO(YOLO_CKPT)
    # Ultralytics handles everything: PT → ONNX → TFLite
    export_path = model.export(format="tflite", imgsz=YOLO_IMG_SIZE)
    print(f"  ✓ YOLO TFLite saved to: {export_path}")

    # Copy to our output dir
    import shutil
    tflite_files = list(Path(export_path).rglob("*.tflite")) if os.path.isdir(export_path) else [Path(export_path)]
    for f in tflite_files:
        dest = os.path.join(OUTPUT_DIR, f"yolo11m_4class.tflite")
        shutil.copy2(str(f), dest)
        print(f"  ✓ Copied to: {dest}")
    return dest


# ══════════════════════════════════════════════════════════════
# 2. EXPORT ResNet50 → TFLite (via ai-edge-torch)
# ══════════════════════════════════════════════════════════════
def export_resnet():
    print("\n[2/2] Exporting ResNet50 to TFLite...")

    # Load the model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(
        torch.load(RESNET_CKPT, map_location="cpu", weights_only=True)
    )
    model.eval()

    # Add softmax at the end so TFLite output is probabilities
    class ResNetWithSoftmax(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return torch.softmax(self.base(x), dim=1)

    model_sm = ResNetWithSoftmax(model)
    model_sm.eval()

    dest = os.path.join(OUTPUT_DIR, "resnet50_4saree.tflite")

    # Try ai-edge-torch first (Google's recommended path)
    try:
        import ai_edge_torch
        sample_input = (torch.randn(1, 3, RESNET_IMG_SIZE, RESNET_IMG_SIZE),)
        edge_model = ai_edge_torch.convert(model_sm, sample_input)
        edge_model.export(dest)
        print(f"  ✓ ResNet50 TFLite saved to: {dest} (via ai-edge-torch)")
        return dest
    except ImportError:
        print("  ⚠ ai-edge-torch not installed, trying ONNX → TFLite route...")
    except Exception as e:
        print(f"  ⚠ ai-edge-torch failed ({e}), trying ONNX → TFLite route...")

    # Fallback: PyTorch → ONNX → TFLite via onnx + onnx2tf
    try:
        import onnx
        onnx_path = os.path.join(OUTPUT_DIR, "resnet50_4saree.onnx")
        dummy = torch.randn(1, 3, RESNET_IMG_SIZE, RESNET_IMG_SIZE)

        torch.onnx.export(
            model_sm, dummy, onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            opset_version=13,
        )
        print(f"  ✓ ONNX exported to: {onnx_path}")

        # Convert ONNX → TFLite using onnx2tf
        import subprocess
        tf_dir = os.path.join(OUTPUT_DIR, "resnet50_tf")
        subprocess.run([
            sys.executable, "-m", "onnx2tf",
            "-i", onnx_path,
            "-o", tf_dir,
            "-oiqt",  # output int8 quantized tflite too
        ], check=True)

        # Find the float32 tflite
        for f in Path(tf_dir).rglob("*float32.tflite"):
            import shutil
            shutil.copy2(str(f), dest)
            print(f"  ✓ ResNet50 TFLite saved to: {dest} (via onnx2tf)")
            return dest

        print("  ✗ Could not find float32 tflite in onnx2tf output")
        return None

    except ImportError as e:
        print(f"\n  ✗ Missing dependency: {e}")
        print("  Install one of:")
        print("    pip install ai-edge-torch")
        print("    pip install onnx onnx2tf")
        return None


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          EXPORT MODELS TO TFLITE                       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    yolo_path   = export_yolo()
    resnet_path = export_resnet()

    print("\n" + "═" * 58)
    print("  SUMMARY")
    print("═" * 58)
    print(f"  YOLO   TFLite: {yolo_path}")
    print(f"  ResNet TFLite: {resnet_path}")
    print(f"  Output dir:    {OUTPUT_DIR}/")
    print("═" * 58)
