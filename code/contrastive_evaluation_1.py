import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Subset

# Import your encoders (Assuming these are available in your project)
from models.resnet50encoder import ResNet50Encoder
from models.efficientnetb4encoder import EfficientNetB4Encoder
from models.xceptionencoder import XceptionEncoder

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from pathlib import Path

class OriginalsOnlyDataset(VisionDataset):
    """
    A custom dataset that mimics ImageFolder but ONLY loads files 
    that start with 'orig_', ignoring all augmented versions.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        # Iterate through directories and strictly grab 'orig_' files
        for cls_name in self.classes:
            cls_dir = Path(root) / cls_name
            if cls_dir.is_dir():
                # Glob specifically for the original prefix
                for img_path in cls_dir.glob("orig_*"): 
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = default_loader(path) # Loads image as RGB PIL
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class PrototypicalEvaluator:
    """
    Evaluates a pretrained encoder using Prototypical Network logic.
    Aligns with SupCon by strictly using L2-normalized embeddings.
    """
    def __init__(self, encoder, device):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device
        self.prototypes = None
        self.classes = None

    @torch.no_grad()
    def compute_prototypes(self, support_loader):
        print("Computing class prototypes from the Support Set...")
        embeddings_list = []
        labels_list = []

        # 1. Extract all embeddings from the support set
        for images, labels in tqdm(support_loader, desc="Extracting Support Features"):
            images = images.to(self.device)
            embeddings = self.encoder(images)
            
            # MUST L2 Normalize to match SupCon training behavior
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())

        all_embeddings = torch.cat(embeddings_list)
        all_labels = torch.cat(labels_list)
        
        self.classes = torch.unique(all_labels)
        num_classes = len(self.classes)
        embed_dim = all_embeddings.shape[1]
        
        self.prototypes = torch.zeros((num_classes, embed_dim))

        # 2. Calculate the mean embedding (prototype) for each class
        for i, cls in enumerate(self.classes):
            cls_mask = (all_labels == cls)
            cls_embeddings = all_embeddings[cls_mask]
            
            # Mean over all embeddings for this class
            mean_embed = cls_embeddings.mean(dim=0)
            
            # Re-normalize the resulting prototype so it sits on the unit sphere
            self.prototypes[i] = F.normalize(mean_embed.unsqueeze(0), p=2, dim=1).squeeze(0)
            
        self.prototypes = self.prototypes.to(self.device)
        print(f"Computed {num_classes} prototypes of dimension {embed_dim}.")

    @torch.no_grad()
    def evaluate(self, query_loader):
        if self.prototypes is None:
            raise ValueError("Prototypes not computed. Call compute_prototypes() first.")
            
        print("Evaluating Query Set against Prototypes...")
        
        all_preds = []
        all_labels = []

        for images, labels in tqdm(query_loader, desc="Evaluating"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Extract and normalize query embeddings
            embeddings = self.encoder(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Compute Cosine Similarity
            similarities = torch.matmul(embeddings, self.prototypes.T)
            
            # Predict
            _, predicted_indices = torch.max(similarities, dim=1)
            predictions = self.classes.to(self.device)[predicted_indices]
            
            # Store for sklearn metrics
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate rich metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
            'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100,
            'f1_score': f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        }
        
        print(f"Results -> Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1_score']:.2f}%")
        return metrics


def run_evaluation(model_name: str, weights_path: str, support_dir: str, query_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n--- Evaluating {model_name} on {device} ---")

    # 1. Initialize the correct base encoder & image size
    if model_name == 'resnet50':
        encoder = ResNet50Encoder()
        img_size = 224
    elif model_name == 'efficientnetb4':
        encoder = EfficientNetB4Encoder()
        img_size = 380
    elif model_name == 'xception':
        encoder = XceptionEncoder()
        img_size = 299
    else:
        raise ValueError("Invalid model name.")

    # 2. Load the trained contrastive weights
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Skipping.")
        return None # Return None instead of just returning so the CSV writer handles it cleanly

    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")

    # 3. Define Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Prepare Support Loader (Fast memory loading)
    support_dataset = OriginalsOnlyDataset(root=support_dir, transform=transform)
    support_loader = DataLoader(support_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Prepare Query Loader (Balanced 10% Subset)
    full_query_dataset = datasets.ImageFolder(root=query_dir, transform=transform)
    
    # Logic to get the first N images per class
    images_per_class = 900 # 900 per class * 10 classes = 9000 images total (~10% of CINIC-10)
    class_counts = {}
    subset_indices = []
    
    for idx, (_, class_idx) in enumerate(full_query_dataset.samples):
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        if class_counts[class_idx] <= images_per_class:
            subset_indices.append(idx)
            
    print(f"Sub-sampled Validation Set: {len(subset_indices)} images total.")
    query_subset = Subset(full_query_dataset, subset_indices)

    # Use the subset with fast memory loading
    query_loader = DataLoader(query_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 6. Run Prototypical Evaluation
    evaluator = PrototypicalEvaluator(encoder, device)
    evaluator.compute_prototypes(support_loader)
    metrics = evaluator.evaluate(query_loader)
    
    return metrics

if __name__ == "__main__":
    # --- Configuration ---
    EXPERIMENTS_DIR = Path("./data/augmented_experiments") 
    QUERY_DATA_DIR = '/kaggle/input/cinic10/valid/'                          
    WEIGHTS_DIR = Path("./pretrained_encoders_contrastive")
    RESULTS_CSV = "results/evaluation_results.csv" # <-- Where the results will be saved!
    
    MODELS_TO_TEST = ['resnet50', 'efficientnetb4', 'xception']
    
    if not EXPERIMENTS_DIR.exists():
        raise FileNotFoundError(f"Cannot find experiments directory: {EXPERIMENTS_DIR}")
        
    dataset_folders = [f for f in EXPERIMENTS_DIR.iterdir() if f.is_dir()]
    print(f"Found {len(dataset_folders)} experimental datasets. Starting evaluation...")

    # Open the CSV file and write the headers
    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])

        for dataset_path in dataset_folders:
            dataset_name = dataset_path.name 
            print(f"\n{'='*50}")
            print(f"Evaluating Experiment: {dataset_name}")
            print(f"{'='*50}")
            
            current_support_dir = str(dataset_path) 
            
            for model_name in MODELS_TO_TEST:
                weights_file = WEIGHTS_DIR / f"{model_name}_{dataset_name}.pth"
                
                if not weights_file.exists():
                    print(f"⏩ Skipping {model_name}: Weights not found at {weights_file}")
                    continue
                    
                try:
                    # Run evaluation now returns a dictionary of metrics
                    metrics = run_evaluation(
                        model_name=model_name,
                        weights_path=str(weights_file),
                        support_dir=current_support_dir, 
                        query_dir=QUERY_DATA_DIR         
                    )
                    
                    if metrics:
                        # Write the row to the CSV and flush it so it saves immediately
                        writer.writerow([
                            dataset_name, 
                            model_name, 
                            f"{metrics['accuracy']:.2f}", 
                            f"{metrics['precision']:.2f}", 
                            f"{metrics['recall']:.2f}", 
                            f"{metrics['f1_score']:.2f}"
                        ])
                        f.flush() 
                        
                except Exception as e:
                    print(f"❌ Failed to evaluate {model_name} on {dataset_name}: {e}")
                    
    print(f"\n✅ All done! Results saved to {RESULTS_CSV}")