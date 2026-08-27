# Models

Tracked PyTorch checkpoints for the ResNet50 branch of the saree classifier.

| File | Description |
|---|---|
| `resnet50_4saree_best.pt` | Best ResNet50 checkpoint (lowest validation loss during fine-tuning) — this is the one used for evaluation, ensembling, and TFLite export |
| `resnet50_4saree_final.pt` | ResNet50 checkpoint from the final training epoch, kept for reference |
| `calibration_image_sample_data_20x128x128x3_float32.npy` | Calibration sample data generated during TFLite export (local-only, gitignored via `*.npy`) |

The YOLO11m counterpart lives in `../runs/classify/YOLO11m_4class4/weights/` alongside its training run.

Checkpoints are written by `../training/train_resnet_finetune.py` and consumed by `evaluate_resnet.py`, `ensemble_compare.py`, and `export_to_tflite.py`.
