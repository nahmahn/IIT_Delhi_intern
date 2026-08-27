# Results

Evaluation plots produced by the scripts in `../training/`.

## ResNet50 evaluation

| File | Description |
|---|---|
| `resnet_evaluation_matrix.png` | Confusion matrix for the ResNet50 checkpoint on the test split (from `evaluate_resnet.py`) |

## Ensemble comparison (from `ensemble_compare.py`)

Each plot exists in two variants: `raw_*` (evaluated on the raw `data/test` images) and `patched_*` (evaluated on the patched `data_patched/test` images).

| File pair | Description |
|---|---|
| `*_confusion_comparison.png` | Side-by-side confusion matrices: ResNet50 vs YOLO11m vs ensemble |
| `*_per_class_comparison.png` | Per-class accuracy for each model and the ensemble |
| `*_ensemble_improvement.png` | Accuracy gained (or lost) by the ensemble over the individual models |
| `*_disagreement.png` | Cases where the two models disagree and how the ensemble resolves them |
