import os
import torch
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score

def compare_runs():
    runs_dir = r'c:\Users\namja\Downloads\textile_design\runs\classify'
    test_dir = r'c:\Users\namja\Downloads\textile_design\data_patched\test'
    output_file = 'model_comparison_results_patched.csv'
    
    # Identify all models
    model_paths = []
    for run in os.listdir(runs_dir):
        best_pt = os.path.join(runs_dir, run, 'weights', 'best.pt')
        if os.path.exists(best_pt):
            model_paths.append((run, best_pt))
    
    if not model_paths:
        print("No models found in runs/classify/*/weights/best.pt")
        return

    print(f"Found {len(model_paths)} models to compare.")
    
    # Prepare test data
    y_true_labels = []
    image_paths = []
    class_names_ref = ["baluchari", "maheshwari", "negammam", "phulkari"]
    
    for class_name in class_names_ref:
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.exists(class_folder):
            continue
        for img in os.listdir(class_folder):
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_paths.append(os.path.join(class_folder, img))
                y_true_labels.append(class_name)

    if not image_paths:
        print(f"No test images found in {test_dir}")
        return

    results = []

    for run_name, model_path in model_paths:
        print(f"\nEvaluating Model: {run_name}")
        try:
            model = YOLO(model_path)
            model_class_names = [v.lower() for v in model.names.values()]
            
            y_pred_labels = []
            
            for img_path in tqdm(image_paths, desc=f"Predicting {run_name}"):
                res = model(img_path, verbose=False)[0]
                pred_idx = res.probs.top1
                pred_label = model.names[pred_idx].lower()
                y_pred_labels.append(pred_label)
            
            # Calculate Metrics
            overall_acc = accuracy_score(y_true_labels, y_pred_labels)
            
            # Per-class recall
            recalls = {}
            for cls in class_names_ref:
                # Filter indices for this specific class
                cls_indices = [i for i, label in enumerate(y_true_labels) if label == cls]
                if not cls_indices:
                    recalls[f"{cls}_recall"] = 0.0
                    continue
                
                cls_true = [y_true_labels[i] for i in cls_indices]
                cls_pred = [y_pred_labels[i] for i in cls_indices]
                
                correct = sum(1 for t, p in zip(cls_true, cls_pred) if t == p)
                recalls[f"{cls}_recall"] = correct / len(cls_indices)

            model_results = {
                "Model": run_name,
                "Accuracy": overall_acc,
                **recalls
            }
            results.append(model_results)
            
        except Exception as e:
            print(f"Error evaluating {run_name}: {e}")

    # Sort results by Accuracy
    results.sort(key=lambda x: x['Accuracy'], reverse=True)

    # Display Table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY (on data_patched/test)")
    print("="*80)
    # Format percentages for display
    display_df = df.copy()
    for col in display_df.columns:
        if col != "Model":
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    print(display_df.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    compare_runs()
