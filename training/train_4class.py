import os
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO

BASE_WEIGHTS = "yolo11m-cls.pt"
DATASET_PATH = "data_patched"
OUTPUT_NAME  = "YOLO11m_4class_v4"

EPOCHS   = 75   # Validation loss peaks early
PATIENCE = 15   # Enforce early stopping
IMGSZ    = 640
BATCH    = 16
DEVICE   = 0


def main():
    global DEVICE

    print("=" * 60)
    print("4-Class YOLO11m — v2 (Baluchari Fix)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU!")
        DEVICE = "cpu"

    # YOLO will automatically download standard weights like yolo11m-cls.pt if missing

    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATASET_PATH, split)
        classes = sorted(os.listdir(split_path))
        counts = {c: len(os.listdir(os.path.join(split_path, c))) for c in classes}
        print(f"{split}: {counts}")

    torch.cuda.empty_cache()
    model = YOLO(BASE_WEIGHTS)

    model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        name=OUTPUT_NAME,
        workers=0,          # Windows requirement

        # Unfreeze all layers to allow the model to build a distinct
        # feature space for all 4 classes simultaneously from the ImageNet base.
        freeze=0,

        # Standard LR since we are starting from ImageNet weights (not fine-tuning)
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,

        # Softens overconfident wrong predictions
        label_smoothing=0.1,

        dropout=0.3,

        # Augmentation (to handle variations in live testing)
        flipud=0.5,
        fliplr=0.5,
        degrees=15,         # textiles are rotationally valid
        scale=0.5,          # simulate different camera zoom distances

        optimizer="AdamW",
        weight_decay=0.0005,

        plots=False,
        save=True,
        val=True,
        verbose=True,
    )

    print(f"\nDone. Best weights: runs/classify/{OUTPUT_NAME}/weights/best.pt")


if __name__ == "__main__":
    freeze_support()
    main()