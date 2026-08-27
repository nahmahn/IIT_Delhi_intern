import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import random
from PIL import Image

# ===============================
# CONFIG
# ===============================
MODEL_PATH  = "runs/classify/YOLO11m_4class_v4/weights/best.pt"
DATASET_PATH = "data_patched"
TEST_DIR    = os.path.join(DATASET_PATH, "test")
CLASS_NAMES = ["baluchari", "maheshwari", "negammam", "phulkari"]

OUTPUT_NAME = "confusion_matrix.png"
WRONG_DIR   = "wrong_predictions"

# ===============================
# SETUP
# ===============================
if os.path.exists(WRONG_DIR):
    shutil.rmtree(WRONG_DIR)
os.makedirs(WRONG_DIR)

model = YOLO(MODEL_PATH)

y_true = []
y_pred = []

# 🔥 extra tracking
all_confidences = []
all_margins = []

print("=" * 60)
print("DEBUG EVALUATION + VISUALIZATION")
print("=" * 60)

# ===============================
# INFERENCE
# ===============================
for class_idx, class_name in enumerate(CLASS_NAMES):
    class_folder = os.path.join(TEST_DIR, class_name)
    images = os.listdir(class_folder)

    for img_name in tqdm(images, desc=class_name):
        img_path = os.path.join(class_folder, img_name)

        results = model(img_path, verbose=False)[0]
        probs = results.probs.data.cpu().numpy()

        top1 = int(np.argmax(probs))
        confidence = probs[top1]

        # 🔥 margin = top1 - top2
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]

        all_confidences.append(confidence)
        all_margins.append(margin)

        # 🔥 top3
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [(CLASS_NAMES[i], probs[i]) for i in top3_idx]

        y_true.append(class_idx)
        y_pred.append(top1)

        # 🚨 WRONG CASES
        if top1 != class_idx:
            save_path = os.path.join(
                WRONG_DIR,
                f"{class_name}_as_{CLASS_NAMES[top1]}_{confidence:.2f}_{img_name}"
            )
            shutil.copy(img_path, save_path)

            print("\n❌ WRONG:")
            print(f"True: {class_name}")
            print(f"Pred: {CLASS_NAMES[top1]}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Margin: {margin:.2f}")
            print(f"Top3: {top3}")

            if confidence > 0.9:
                print("🔥 HIGH CONFIDENCE ERROR")

# ===============================
# METRICS
# ===============================
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

overall_acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {overall_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix")
plt.savefig(OUTPUT_NAME)
plt.close()

# ===============================
# CONFIDENCE HISTOGRAM
# ===============================
plt.figure()
plt.hist(all_confidences, bins=20)
plt.title("Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.savefig("confidence_hist.png")
plt.close()

# ===============================
# MARGIN HISTOGRAM
# ===============================
plt.figure()
plt.hist(all_margins, bins=20)
plt.title("Top1-Top2 Margin Distribution")
plt.xlabel("Margin")
plt.ylabel("Count")
plt.savefig("margin_hist.png")
plt.close()

# ===============================
# PER-CLASS CONFUSION BREAKDOWN
# ===============================
print("\nPer-class confusion analysis:")

for i, class_name in enumerate(CLASS_NAMES):
    idx = (y_true == i)
    preds = y_pred[idx]

    counts = np.bincount(preds, minlength=len(CLASS_NAMES))

    print(f"\n{class_name}:")
    for j, count in enumerate(counts):
        print(f"  → {CLASS_NAMES[j]}: {count}")

# ===============================
# WRONG IMAGE GRID
# ===============================
wrong_images = os.listdir(WRONG_DIR)

if len(wrong_images) > 0:
    sample = random.sample(wrong_images, min(9, len(wrong_images)))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for ax, img_name in zip(axes.flatten(), sample):
        img_path = os.path.join(WRONG_DIR, img_name)
        img = Image.open(img_path)

        ax.imshow(img)

        parts = img_name.split("_")
        title = f"{parts[0]} → {parts[2]}"
        ax.set_title(title)

        ax.axis("off")

    plt.tight_layout()
    plt.savefig("wrong_samples.png")
    plt.close()

print("\nSaved confusion matrix:", OUTPUT_NAME)
print("Saved wrong predictions in:", WRONG_DIR)
print("Saved confidence histogram: confidence_hist.png")
print("Saved margin histogram: margin_hist.png")
print("Saved wrong samples grid: wrong_samples.png")