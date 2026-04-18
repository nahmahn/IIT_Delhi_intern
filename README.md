# Indian Textile Art Classifier — Engineer Handover

> Classification of three Indian embroidery styles (Maheshwari, Negammam, Phulkari) using YOLO11. Trained in Python, exported to TFLite float16, deployed in a native Android app with real-time camera inference.

---

## Project Overview

Five variants of the YOLO11 image classification architecture were trained and benchmarked on a curated dataset of three Indian textile art forms. YOLO11m was selected as the best model (~92% test accuracy) and exported to TFLite float16 for Android deployment. The Android app runs fully on-device — no server, no internet required.

---

## Classes

| Class | Description |
|---|---|
| Maheshwari | Fine silk-cotton sarees from Maheshwar, Madhya Pradesh — distinctive border patterns and reversal weave |
| Negammam | Traditional embroidery from Tamil Nadu — bold geometric motifs on cotton fabric |
| Phulkari | Floral needlework from Punjab — vibrant silk threads on handspun cotton (khaddar) |

---

## Where Everything Lives

```
textile_design/
│
├── model.ipynb
│   Training loop for all 5 YOLO11 variants + benchmark evaluation
│   + per-class confusion matrix generation on the test split
│
├── tflit.ipynb
│   Export pipeline: best.pt → ONNX (opset 12) → TF SavedModel → TFLite float16
│   Also contains standalone TFLite inference test against the test split
│
├── selected_18_02/               ← dataset root (referenced in both notebooks)
│   ├── train/
│   ├── val/
│   └── test/
│
├── YOLO11m_benchmark/            ← training run output for the deployed model
│   └── weights/
│       ├── best.pt               ← ✅ BEST TRAINED MODEL (PyTorch weights)
│       └── best_saved_model/
│           └── best_float16.tflite   ← ✅ DEPLOYED MODEL (TFLite float16)
│
├── results_n_weights/            ← benchmark plots
│   ├── YOLO11n_test_confusion_matrix.png
│   ├── YOLO11s_test_confusion_matrix.png
│   ├── YOLO11m_test_confusion_matrix.png   ← confusion matrix for deployed model
│   ├── YOLO11l_test_confusion_matrix.png
│   ├── YOLO11x_test_confusion_matrix.png
│   └── YOLO11_benchmark_comparison.png     ← bar chart: test accuracy across all 5 models
│
└── AndroidApp/                   ← Android Studio project
    └── ...                       ← TFLite model is bundled as an asset inside here
```

> **Key paths:**
> - Best PyTorch weights: `textile_design/YOLO11m_benchmark/weights/best.pt`
> - Deployed TFLite model: `textile_design/YOLO11m_benchmark/weights/best_saved_model/best_float16.tflite`

---

## Workflow Diagram

![Project workflow](workflow.svg)

---

## Training Details

Framework: Ultralytics YOLO11 (`ultralytics` Python package), image classification task (`-cls`).

All five scales were trained independently on the same `selected_18_02` dataset split. Training runs are saved under `textile_design/runs/classify/` — the relevant one is `YOLO11m_benchmark/`.

`model.ipynb` handles training, manual test-set inference, confusion matrix generation, and the benchmark comparison across all five models. The variable `dataset_path` in that notebook is set to `"selected_18_02"` — the dataset must be present at that relative path when running.

---

## Benchmark Results

| Model | Test Accuracy | Notes |
|---|---|---|
| YOLO11n | ~90% | Smallest; generalises well |
| YOLO11s | ~90% | Small |
| **YOLO11m** | **~92%** | **Selected — best accuracy/size tradeoff** |
| YOLO11l | ~91% | Large; marginal gain over YOLO11m |
| YOLO11x | ~84% | Overfits on this dataset size |

### Confusion Matrix — YOLO11m (deployed model)

| True ↓ / Predicted → | Maheshwari | Negammam | Phulkari |
|---|---|---|---|
| Maheshwari | **28** | 2 | 1 |
| Negammam | 1 | **27** | 3 |
| Phulkari | 0 | 0 | **31** |

Phulkari is classified perfectly (31/31). Negammam has minor leakage into Phulkari (3 cases). Maheshwari is strong with minor confusion. Overall solid across all three classes.

---

## Export Pipeline

Handled entirely in `tflit.ipynb`. The path to the source weights is hardcoded in that notebook:

```
pt_model_path = "/home/kniting/textile_design/runs/classify/YOLO11m_benchmark/weights/best.pt"
```

The pipeline runs:

```
best.pt
  → model.export(format="onnx", opset=12)    →  model.onnx
  → onnx_tf backend                           →  tf_model/  (TF SavedModel)
  → tf.lite.TFLiteConverter (float16)         →  best_float16.tflite
```

The notebook also contains a standalone TFLite inference loop that runs `best_float16.tflite` against the test split and regenerates the confusion matrix — useful for validating the exported model without Android.

---

## Android App

Built in Android Studio. The model `best_float16.tflite` is bundled as an asset inside the app. The TFLite interpreter runs on-device — no network calls, no backend. Camera frames are preprocessed to 640×640 before inference. The app displays the predicted class label and confidence score in real time.

The Android project is not in this repo due to size — check the original machine at `~/textile_design/AndroidApp/` or equivalent.

---

## Key Files at a Glance

| File | Purpose |
|---|---|
| `model.ipynb` | Train all 5 YOLO11 variants, evaluate on test set, generate plots |
| `tflit.ipynb` | Export YOLO11m to TFLite float16, validate TFLite inference |
| `textile_design/YOLO11m_benchmark/weights/best.pt` | Best trained PyTorch model |
| `textile_design/YOLO11m_benchmark/weights/best_saved_model/best_float16.tflite` | Deployed TFLite model |
| `selected_18_02/` | Dataset root (train / val / test) |
| `results_n_weights/` | Confusion matrices + benchmark bar chart |

---

## Dependencies

```
ultralytics
scikit-learn
tensorflow
onnx
onnxruntime
tf2onnx
onnx-tf
opencv-python
matplotlib
```
