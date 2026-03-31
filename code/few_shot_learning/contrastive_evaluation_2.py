import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.efficientnetb4encoder import EfficientNetB4Encoder
from contrastive_evaluation_1 import OriginalsOnlyDataset, PrototypicalEvaluator




def run_evaluation(weights_path: str, support_dir: str, query_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # if model_name == 'resnet50':
    #     encoder = ResNet50Encoder()
    #     img_size = 224
    # elif model_name == 'efficientnetb4':
    encoder = EfficientNetB4Encoder()
    img_size = 380
    # elif model_name == 'xception':
    #     encoder = XceptionEncoder()
    #     img_size = 299


    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    support_dataset = OriginalsOnlyDataset(root=support_dir, transform=transform)
    support_loader = DataLoader(support_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

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

    return metrics

if __name__ == "__main__":
    EXPERIMENTS_DIR = Path("./data/augmented_experiments") 
    QUERY_DATA_DIR = 'data/valid/'                          
    WEIGHTS_DIR = Path("./pretrained_encoders_contrastive")
    RESULTS_CSV = "results/evaluation_results_phase2.csv"
    
    TARGET_DATASETS = [
        "10shot_50aug_standard_rotate", 
        "10shot_50aug_standard_mixed", 
        "10shot_50aug_advanced_mixed"
    ]
    L_VALUES = [0, 5, 15, 30, 50]

    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Model', 'L_Value', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])

        for dataset_name in TARGET_DATASETS:
            dataset_path = EXPERIMENTS_DIR / dataset_name
            if not dataset_path.exists(): 
                print(f"Directory {dataset_path} not found. Skipping.")
                continue
            
            print(f"\n{'='*50}\nEvaluating Experiment: {dataset_name}\n{'='*50}")
            current_support_dir = str(dataset_path) 
            
            for l_val in L_VALUES:
                print(f"\nEvaluating with L={l_val}...")
                weights_file = WEIGHTS_DIR / f"efficientnetb4_{dataset_name}_l{l_val}.pth"
                
                if not weights_file.exists():
                    print(f"Skipping l={l_val}: Weights not found at {weights_file}")
                    continue
                    
                try:
                    metrics = run_evaluation(
                        weights_path=str(weights_file),
                        support_dir=current_support_dir, 
                        query_dir=QUERY_DATA_DIR         
                    )
                    
                    if metrics:
                        writer.writerow([
                            dataset_name, 
                            'efficientnetb4', 
                            l_val, 
                            f"{metrics['accuracy']:.2f}", 
                            f"{metrics['precision']:.2f}", 
                            f"{metrics['recall']:.2f}", 
                            f"{metrics['f1_score']:.2f}"
                        ])
                        f.flush() 
                        
                except Exception as e:
                    print(f"Failed: {e}")
                    