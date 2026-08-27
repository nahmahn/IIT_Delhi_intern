# Classification

Everything related to the 4-class saree classifier (Baluchari, Maheshwari, Negamam, Phulkari): training, evaluation, ensembling, and TFLite export for the mobile app.

## Contents

| Folder | Purpose |
|---|---|
| [training/](training/) | All training, evaluation, comparison, and export scripts |
| [models/](models/) | Tracked ResNet50 checkpoints |
| [results/](results/) | Evaluation plots: confusion matrices, ensemble comparisons |
| [runs/](runs/) | Ultralytics YOLO training runs (`YOLO11m_4class4` is the deployed one) |
| `environment.yml` | Conda environment ("saree") with all training dependencies |

## Pipeline at a glance

1. Raw images in `data/` are patched into `data_patched/` (`training/patch.py`).
2. YOLO11m and ResNet50 are trained on the patched dataset.
3. Both are evaluated individually and as a weighted ensemble.
4. Both are exported to TFLite and bundled into the Flutter app (`app/tana_app/assets/models/`).

## Setup

```bash
conda env create -f environment.yml
conda activate saree
```

Run every script with this `classification/` folder as the working directory:

```bash
python training/train_4class.py
```

## Local-only folders

The following live here on the development machine but are gitignored (datasets and derived artifacts): `data/`, `data_patched/`, `random_16_02/`, `tflite_models/`, `YOLO11m_benchmark/`, `wrong_predictions/`.
