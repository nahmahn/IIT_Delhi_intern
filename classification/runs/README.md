# Runs

Ultralytics training output for the YOLO11m classification runs, under `classify/`. Each folder contains `args.yaml` (the exact training configuration); completed runs also contain weights, curves, and confusion matrices.

| Run | Notes |
|---|---|
| `YOLO11m_4class` — `YOLO11m_4class3` | Early attempts (config snapshots only) |
| **`YOLO11m_4class4`** | **The deployed run.** Fine-tuned from the earlier 3-class benchmark weights; best validation top-1 accuracy ~93.5%. Contains `weights/best.pt`, ONNX and TFLite exports under `weights/best_saved_model/`, and a full write-up in [model_report.md](classify/YOLO11m_4class4/model_report.md) |
| `YOLO11m_4class_v2` — `YOLO11m_4class_v4` | Iterations with weights kept for comparison via `../training/compare_models.py` |

The key artifact is `classify/YOLO11m_4class4/weights/best.pt`, which feeds the ensemble and the TFLite export used by the mobile app.
