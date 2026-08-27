# Training scripts

Scripts for training, evaluating, ensembling, and exporting the 4-class saree classifiers. Run all of them with `classification/` (the parent folder) as the working directory — paths inside the scripts are relative to it.

## Data preparation

| Script | Purpose |
|---|---|
| `patch.py` | Extracts patches from the raw images (`data/` to `data_patched/`) so models train on fabric detail rather than whole photos |

## Training

| Script | Purpose |
|---|---|
| `train_4class.py` | Trains YOLO11m (classification head) on `data_patched/`; output goes to `runs/classify/` |
| `train_resnet_finetune.py` | Fine-tunes a torchvision ResNet50; writes `models/resnet50_4saree_best.pt` and `models/resnet50_4saree_final.pt` |

## Evaluation

| Script | Purpose |
|---|---|
| `evaluate_4class.py` | Evaluates the YOLO11m checkpoint on the test split; copies misclassified images to `wrong_predictions/` |
| `evaluate_resnet.py` | Evaluates the ResNet50 checkpoint; writes `results/resnet_evaluation_matrix.png` |
| `compare_models.py` | Benchmarks every run under `runs/classify/` on the test split; writes a comparison CSV |
| `ensemble_compare.py` | Compares ResNet50 vs YOLO11m vs their weighted ensemble on both raw and patched test sets; plots go to `results/` |

## Export and TFLite validation

| Script | Purpose |
|---|---|
| `export_to_tflite.py` | Exports both models to TFLite (float16/float32) into `tflite_models/` for the mobile app |
| `ensemble_tflite.py` | Re-runs the ensemble evaluation using the exported TFLite models to confirm parity |
| `diagnose_tflite.py` | Compares .pt vs .tflite predictions image-by-image to debug accuracy drift after export |

The ensemble weights found here (YOLO 0.85, ResNet 0.15) are hardcoded into the Flutter app's `classifier_service.dart`.
