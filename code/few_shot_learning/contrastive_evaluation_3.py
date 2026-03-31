import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets import VisionDataset
from tqdm import tqdm
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.efficientnetb4encoder import EfficientNetB4Encoder
from few_shot_phase_3 import VariableKDataset
from contrastive_evaluation_1 import PrototypicalEvaluator



def run_k_evaluation(k_val: int, weights_path: str, support_dir: str, query_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n--- Evaluating EfficientNet-B4 | K={k_val} on {device} ---")

    encoder = EfficientNetB4Encoder()
    img_size = 380

    encoder.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    support_dataset = VariableKDataset(root=support_dir, k_max=k_val, transform=transform)
    support_loader = DataLoader(support_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    full_query_dataset = datasets.ImageFolder(root=query_dir, transform=transform)
    images_per_class = 900 
    class_counts = {}
    subset_indices = []
    
    for idx, (_, class_idx) in enumerate(full_query_dataset.samples):
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        if class_counts[class_idx] <= images_per_class:
            subset_indices.append(idx)
            
    query_subset = Subset(full_query_dataset, subset_indices)
    query_loader = DataLoader(query_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    evaluator = PrototypicalEvaluator(encoder, device)
    evaluator.compute_prototypes(support_loader)
    metrics = evaluator.evaluate(query_loader)
    
    print(f"Results for K={k_val} - Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1_score']:.2f}%")
    return metrics

if __name__ == "__main__":
    MASTER_SUPPORT_DIR = Path("./data/augmented_experiments/50shot_0aug_baseline") 
    QUERY_DATA_DIR = './data/valid'                          
    WEIGHTS_DIR = Path("./pretrained_encoders_contrastive")
    RESULTS_CSV = "results/evaluation_results_k_experiments.csv" 
    
    K_VALUES = [5, 10, 15, 20, 30, 40, 50]

    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['K_Value', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])

        for k_val in K_VALUES:
            print(f"\n{'='*50}")
            print(f"Evaluating Baseline Set with K={k_val}")
            print(f"{'='*50}")
            
            weights_file = WEIGHTS_DIR / f"efficientnetb4_baseline_k{k_val}.pth"
                
            try:
                metrics = run_k_evaluation(
                    k_val=k_val,
                    weights_path=str(weights_file),
                    support_dir=str(MASTER_SUPPORT_DIR), 
                    query_dir=QUERY_DATA_DIR         
                )
                
                if metrics:
                    writer.writerow([
                        k_val, 
                        'efficientnetb4', 
                        f"{metrics['accuracy']:.2f}", 
                        f"{metrics['precision']:.2f}", 
                        f"{metrics['recall']:.2f}", 
                        f"{metrics['f1_score']:.2f}"
                    ])
                    f.flush() 
                    
            except Exception as e:
                print(f"Failed: {e}")
            