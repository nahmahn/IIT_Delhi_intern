import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
DATA_DIR          = "data/test" # Falls back to data_patched/val if test doesn't exist
CHECKPOINT_PATH   = "resnet50_4saree_best.pt"
BATCH_SIZE        = 32
NUM_CLASSES       = 4
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    print(f"Loading ResNet50 and checkpoint from {CHECKPOINT_PATH}...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    # Replace head to match fine-tuned model
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
        print("[OK] Checkpoint loaded successfully.")
    else:
        print(f"[!] Warning: Checkpoint {CHECKPOINT_PATH} not found. Running with untrained weights!")
    
    return model.to(DEVICE)

def main():
    # Setup paths 
    eval_dir = DATA_DIR
    if not os.path.exists(eval_dir):
        fallback_dir = "data_patched/val"
        if os.path.exists(fallback_dir):
            print(f"[!] The directory {eval_dir} was not found. Falling back to val set: {fallback_dir}")
            eval_dir = fallback_dir
        else:
            print(f"[!] Error: Data directory not found. Please ensure 'data_patched/test' or 'val' exists.")
            return

    # Standard ResNet evaluation transforms
    transform = transforms.Compose([
        transforms.Resize(672),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    class_names = dataset.classes
    
    print(f"\nEvaluating on {len(dataset)} images across {len(class_names)} classes.")
    print(f"Classes: {class_names}\n")

    model = build_model()
    model.eval()
    
    all_preds = []
    all_labels = []

    # Run Inference
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Scoring dataset"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # -----------------------------------------------------
    # Metrics Calculation
    # -----------------------------------------------------
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    # Sklearn's classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # Generate and save Confusion Matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('ResNet50 Saree Classification - Confusion Matrix', pad=15)
    plt.tight_layout()
    
    plot_filepath = 'resnet_evaluation_matrix.png'
    plt.savefig(plot_filepath, dpi=300)
    print(f"\n[OK] Awesome! Saved a visual confusion matrix to: {plot_filepath}")

if __name__ == "__main__":
    main()
