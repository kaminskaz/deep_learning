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
from models.xceptionencoder import XceptionEncoder
from models.resnet50encoder import ResNet50Encoder
from few_shot_phase_3 import VariableKDataset
from contrastive_evaluation_1 import PrototypicalEvaluator


if __name__ == "__main__":
    MASTER_SUPPORT_DIR = Path("./data/augmented_experiments/50shot_0aug_baseline") 
    QUERY_DATA_DIR = 'data/valid/'                          
    RESULTS_CSV = "results/evaluation_results_raw_pretrained_baseline.csv" 
    
    K_VALUES = [5, 10, 15, 20, 30, 40, 50]
    IMG_SIZE = 380

    if not MASTER_SUPPORT_DIR.exists():
        raise FileNotFoundError(f"Missing master support directory: {MASTER_SUPPORT_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


    encoder = XceptionEncoder().to(device)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_query_dataset = datasets.ImageFolder(root=QUERY_DATA_DIR, transform=transform)
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

    with open(RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['K_Value', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])

        for k_val in K_VALUES:
            print(f"\n{'='*50}")
            print(f"Evaluating Raw Pretrained Weights with K={k_val}")
            
            support_dataset = VariableKDataset(root=str(MASTER_SUPPORT_DIR), k_max=k_val, transform=transform)
            support_loader = DataLoader(support_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
                
            try:
                evaluator.compute_prototypes(support_loader)
                metrics = evaluator.evaluate(query_loader)
                
                print(f"Results for K={k_val} -> Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1_score']:.2f}%")
                
                writer.writerow([
                    k_val, 
                    'xception_raw_pretrained', 
                    f"{metrics['accuracy']:.2f}", 
                    f"{metrics['precision']:.2f}", 
                    f"{metrics['recall']:.2f}", 
                    f"{metrics['f1_score']:.2f}"
                ])
                f.flush() 
                    
            except Exception as e:
                print(f"Failed: {e}")
                    