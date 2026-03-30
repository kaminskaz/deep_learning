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

class VariableKSupportDataset(VisionDataset):
    """
    Strictly loads the first 'k_max' original images per class to build prototypes.
    """
    def __init__(self, root, k_max, transform=None):
        super().__init__(root, transform=transform)
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_dir = Path(root) / cls_name
            if cls_dir.is_dir():
                orig_files = sorted(list(cls_dir.glob("orig_*")))
                for orig_path in orig_files[:k_max]:
                    self.samples.append((str(orig_path), self.class_to_idx[cls_name]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class PrototypicalEvaluator:
    def __init__(self, encoder, device):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device
        self.prototypes = None
        self.classes = None

    @torch.no_grad()
    def compute_prototypes(self, support_loader):
        embeddings_list = []
        labels_list = []

        for images, labels in tqdm(support_loader, desc="Extracting Prototypes", leave=False):
            images = images.to(self.device)
            embeddings = F.normalize(self.encoder(images), p=2, dim=1)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())

        all_embeddings = torch.cat(embeddings_list)
        all_labels = torch.cat(labels_list)
        
        self.classes = torch.unique(all_labels)
        num_classes = len(self.classes)
        embed_dim = all_embeddings.shape[1]
        
        self.prototypes = torch.zeros((num_classes, embed_dim))

        for i, cls in enumerate(self.classes):
            cls_mask = (all_labels == cls)
            mean_embed = all_embeddings[cls_mask].mean(dim=0)
            self.prototypes[i] = F.normalize(mean_embed.unsqueeze(0), p=2, dim=1).squeeze(0)
            
        self.prototypes = self.prototypes.to(self.device)

    @torch.no_grad()
    def evaluate(self, query_loader):
        if self.prototypes is None:
            raise ValueError("Prototypes not computed.")
            
        all_preds = []
        all_labels = []

        for images, labels in tqdm(query_loader, desc="Evaluating Query Set", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            embeddings = F.normalize(self.encoder(images), p=2, dim=1)
            
            similarities = torch.matmul(embeddings, self.prototypes.T)
            _, predicted_indices = torch.max(similarities, dim=1)
            predictions = self.classes.to(self.device)[predicted_indices]
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
            'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
            'f1_score': f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        }

if __name__ == "__main__":
    MASTER_SUPPORT_DIR = Path("./data/augmented_experiments/50shot_0aug_baseline") 
    QUERY_DATA_DIR = 'data/valid/'                          
    RESULTS_CSV = "results/evaluation_results_raw_pretrained_baseline.csv" 
    
    K_VALUES = [5, 10, 15, 20, 30, 40, 50]
    IMG_SIZE = 380

    if not MASTER_SUPPORT_DIR.exists():
        raise FileNotFoundError(f"Missing master support directory: {MASTER_SUPPORT_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n--- Initializing Raw Pretrained Xception on {device} ---")

    encoder = XceptionEncoder().to(device)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\nPreparing balanced 10% query set...")
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
    print(f"Query set ready: {len(query_subset)} total images.")

    evaluator = PrototypicalEvaluator(encoder, device)

    with open(RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['K_Value', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])

        for k_val in K_VALUES:
            print(f"\n{'='*50}")
            print(f"Evaluating Raw Pretrained Weights with K={k_val}")
            
            support_dataset = VariableKSupportDataset(root=str(MASTER_SUPPORT_DIR), k_max=k_val, transform=transform)
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
                print(f"Failed to evaluate K={k_val}: {e}")
                    
    print(f"\nAll done! Baseline results saved to {RESULTS_CSV}")