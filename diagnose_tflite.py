"""
Diagnose TFLite inference vs YOLO native inference.
Simulates what the Dart app does vs what YOLO does natively.
This tells us exactly where the accuracy loss is coming from.
"""
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf

# ===========================
# CONFIG
# ===========================
MODEL_PT   = "runs/classify/YOLO11m_4class_v3/weights/best.pt"
MODEL_TFLITE = "runs/classify/YOLO11m_4class_v3/weights/best_saved_model/best_float32.tflite"
DATASET_PATH = "data_patched"
TEST_DIR     = os.path.join(DATASET_PATH, "test")
CLASS_NAMES  = ["baluchari", "maheshwari", "negammam", "phulkari"]
IMG_SIZE     = 640

# ===========================
# LOAD MODELS
# ===========================
print("Loading YOLO .pt model...")
yolo_model = YOLO(MODEL_PT)

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n--- TFLite Input Details ---")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Dtype: {input_details[0]['dtype']}")
print(f"  Quantization: {input_details[0].get('quantization', 'N/A')}")
print(f"  Quantization params: {input_details[0].get('quantization_parameters', 'N/A')}")

print(f"\n--- TFLite Output Details ---")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Dtype: {output_details[0]['dtype']}")


# ===========================
# PREPROCESSING: "DART APP" STYLE
# (squash resize whole image to 640x640)
# ===========================
def preprocess_dart_style(img_path):
    """Simulates what classifier_service.dart does."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))  # squash resize
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 3)


# ===========================
# PREPROCESSING: "YOLO" STYLE
# (resize shortest side, center crop)
# ===========================
def preprocess_yolo_style(img_path):
    """Simulates YOLO's classify preprocessing."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Resize shortest side to IMG_SIZE, maintain aspect ratio
    scale = IMG_SIZE / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to IMG_SIZE x IMG_SIZE
    left = (new_w - IMG_SIZE) // 2
    top  = (new_h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 3)


# ===========================
# RUN TFLITE
# ===========================
def run_tflite(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]


# ===========================
# EVALUATE ALL THREE METHODS
# ===========================
results = {
    "yolo_native": {"correct": 0, "total": 0},
    "tflite_dart":  {"correct": 0, "total": 0},
    "tflite_yolo":  {"correct": 0, "total": 0},
}

disagreements = []

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(TEST_DIR, class_name)
    images = os.listdir(class_dir)

    print(f"\n--- {class_name} ({len(images)} images) ---")

    for img_name in images:
        img_path = os.path.join(class_dir, img_name)

        # 1. YOLO native (.pt)
        yolo_result = yolo_model(img_path, verbose=False)[0]
        yolo_probs = yolo_result.probs.data.cpu().numpy()
        yolo_pred = int(np.argmax(yolo_probs))

        # 2. TFLite with Dart-style preprocessing (squash resize)
        dart_input = preprocess_dart_style(img_path)
        dart_probs = run_tflite(dart_input)
        dart_pred = int(np.argmax(dart_probs))

        # 3. TFLite with YOLO-style preprocessing (center crop)
        yolo_input = preprocess_yolo_style(img_path)
        yolo_tfl_probs = run_tflite(yolo_input)
        yolo_tfl_pred = int(np.argmax(yolo_tfl_probs))

        results["yolo_native"]["total"] += 1
        results["tflite_dart"]["total"] += 1
        results["tflite_yolo"]["total"] += 1

        if yolo_pred == class_idx:
            results["yolo_native"]["correct"] += 1
        if dart_pred == class_idx:
            results["tflite_dart"]["correct"] += 1
        if yolo_tfl_pred == class_idx:
            results["tflite_yolo"]["correct"] += 1

        # Log disagreements between methods
        if dart_pred != yolo_pred:
            disagreements.append({
                "image": img_name,
                "true": class_name,
                "yolo_native": CLASS_NAMES[yolo_pred],
                "tflite_dart": CLASS_NAMES[dart_pred],
                "tflite_yolo": CLASS_NAMES[yolo_tfl_pred],
                "dart_conf": f"{dart_probs[dart_pred]:.3f}",
                "yolo_conf": f"{yolo_probs[yolo_pred]:.3f}",
            })

# ===========================
# PRINT SUMMARY
# ===========================
print("\n" + "=" * 60)
print("DIAGNOSIS RESULTS")
print("=" * 60)

for method, data in results.items():
    acc = data["correct"] / data["total"] * 100
    print(f"  {method:20s}:  {data['correct']}/{data['total']}  ({acc:.1f}%)")

print(f"\n  Disagreements (dart vs yolo): {len(disagreements)}")

if disagreements:
    print("\n  Sample disagreements:")
    for d in disagreements[:10]:
        print(f"    {d['true']:12s} | YOLO={d['yolo_native']:12s}({d['yolo_conf']}) | "
              f"TFLite-Dart={d['tflite_dart']:12s}({d['dart_conf']}) | "
              f"TFLite-YOLO={d['tflite_yolo']}")

print("\n✅ Done. If tflite_dart accuracy << tflite_yolo, the issue is preprocessing.")
print("   If tflite_yolo accuracy << yolo_native, the issue is in the TFLite export.")
print("   If yolo_native accuracy is already low, the issue is the model itself.")
