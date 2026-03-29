import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from models.resnet50encoder import ResNet50Encoder
from models.efficientnetb4encoder import EfficientNetB4Encoder
from models.xceptionencoder import XceptionEncoder
from contrastive_evaluation_1 import OriginalsOnlyDataset 
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json
import os
import json
import glob
import kagglehub
import sys
import torch
import torch
import torch.nn.functional as F
from itertools import combinations
from torchmetrics import Precision, Recall, F1Score, Accuracy
import pandas as pd
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy
from tqdm import tqdm
from itertools import combinations
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "utils"))

from utils import get_dataloader, seed_everything
from prepare_summary_csv import build_model  

def get_best_models_from_csv(csv_path: str, top_k: int = 1) -> pd.DataFrame:
    """
    Reads the results CSV and selects the best configuration for each architecture
    based on validation accuracy.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['val_accuracy'])
    
    best_models = df.sort_values(by='val_accuracy', ascending=False).groupby('model_name').head(top_k)
    return best_models

def find_model_weights(saved_models_dir: str, row: pd.Series) -> str:
    """
    Reconstructs the expected filename from the CSV row and finds it in the directory.
    """
    model_dir = os.path.join(saved_models_dir, row['model_name'])
    
    search_pattern = f"*experiment_{row['experiment']}*bs{row['batch_size']}_{row['optimizer']}_lr{row['lr']}_reg{row['regularization']}_lambda{row['lambda']}_drop{row['dropout']}.pth"
    
    search_path = os.path.join(model_dir, search_pattern)
    matched_files = glob.glob(search_path)
    
    if not matched_files:
        raise FileNotFoundError(f"Could not find weights for config: {search_pattern} in {model_dir}")
    
    return matched_files[0]

def get_model_predictions(model: torch.nn.Module, data_loader, device: str):
    """
    Runs inference for a single model and returns probabilities, predictions, and targets.
    """
    model.eval()
    model.to(device)
    
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"Predicting", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs.float())
            
            probs = F.softmax(logits, dim=1) 
            _, preds = torch.max(probs, 1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    return torch.cat(all_probs), torch.cat(all_preds), torch.cat(all_targets)

def hard_voting(preds_list: list, probs_list: list):
    """
    Majority class voting. Breaks ties using cumulative probability.
    """
    stacked_preds = torch.stack(preds_list, dim=1)
    sum_probs = torch.stack(probs_list, dim=0).sum(dim=0)
    
    final_preds = []
    for i in range(stacked_preds.shape[0]):
        sample_preds = stacked_preds[i]
        unique_preds, counts = torch.unique(sample_preds, return_counts=True)
        max_count = counts.max()
        
        tied_classes = unique_preds[counts == max_count]
        
        if len(tied_classes) == 1:
            final_preds.append(tied_classes[0].item())
        else:
            tied_probs = sum_probs[i, tied_classes]
            best_tied_idx = torch.argmax(tied_probs)
            final_preds.append(tied_classes[best_tied_idx].item())
            
    return torch.tensor(final_preds)

def soft_voting(probs_list: list):
    """
    Averages the softmax probabilities across all models and takes the argmax.
    """
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0) 
    final_preds = torch.argmax(avg_probs, dim=1)
    return final_preds

def evaluate_predictions(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 10):
    """
    Calculates metrics using torchmetrics.
    """
    precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
    recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro')
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    return {
        "precision": precision(preds, targets).item(),
        "recall": recall(preds, targets).item(),
        "f1_score": f1_score(preds, targets).item(),
        "accuracy": accuracy(preds, targets).item()
    }

# --- DUMMY DATA GENERATORS ---

def generate_dummy_predictions(num_samples: int, num_classes: int):
    """
    Simulates a model's output by generating random logits and converting them to probabilities.
    """
    logits = torch.randn(num_samples, num_classes)
    probs = F.softmax(logits, dim=1)
    _, preds = torch.max(probs, 1)
    return probs, preds

def main():
    # 1. Setup Dummy Parameters
    num_samples = 150
    num_classes = 10
    ensemble_results_path = "dummy_ensemble_results.json"
    
    print(f"Generating dummy test data for {num_samples} samples across {num_classes} classes...\n")
    
    # Generate a single set of ground truth targets so all models are evaluated fairly
    targets = torch.randint(0, num_classes, (num_samples,))
    
    # Define our mock "best models"
    mock_models = [
        {"model_name": "resnet50", "csv_val_acc": 0.85, "weights_path": "dummy/path/res.pth"},
        {"model_name": "efficientnetb4", "csv_val_acc": 0.88, "weights_path": "dummy/path/eff.pth"},
        {"model_name": "xception", "csv_val_acc": 0.86, "weights_path": "dummy/path/xcep.pth"}
    ]

    # 2. Generate Dummy Predictions for each model
    model_outputs = {}
    for config in mock_models:
        model_name = config["model_name"]
        probs, preds = generate_dummy_predictions(num_samples, num_classes)
        
        model_outputs[model_name] = {
            "probs": probs,
            "preds": preds,
            "config": config
        }
        
        # Quick sanity check print
        acc = evaluate_predictions(preds, targets, num_classes)["accuracy"]
        print(f"{model_name} dummy accuracy: {acc:.4f} (expected to be ~0.10 since it's random)")

    # 3. Generate Ensembles
    model_names = list(model_outputs.keys())
    # Create combinations: all 3 together, plus pairwise combinations
    ensemble_combinations = [model_names] + list(combinations(model_names, 2))
    
    ensemble_records = []

    print("\nEvaluating Dummy Ensembles...")
    for combo in ensemble_combinations:
        combo_name = " + ".join(combo)
        print(f"Testing: {combo_name}")
        
        combo_probs = [model_outputs[m]["probs"] for m in combo]
        combo_preds = [model_outputs[m]["preds"] for m in combo]
        
        # Soft Voting
        soft_preds = soft_voting(combo_probs)
        soft_metrics = evaluate_predictions(soft_preds, targets, num_classes)
        
        # Hard Voting
        hard_preds = hard_voting(combo_preds, combo_probs)
        hard_metrics = evaluate_predictions(hard_preds, targets, num_classes)
        
        # Store Record
        record = {
            "ensemble_name": combo_name,
            "protocol": "standard_supervised",
            "models_included": [
                {
                    "architecture": m,
                    "csv_val_acc": model_outputs[m]["config"]["csv_val_acc"],
                    "weights_path": model_outputs[m]["config"]["weights_path"]
                } for m in combo
            ],
            "results": {
                "soft_voting": soft_metrics,
                "hard_voting": hard_metrics
            }
        }
        ensemble_records.append(record)

    # 4. Save Results to JSON
    with open(ensemble_results_path, "w") as f:
        json.dump(ensemble_records, f, indent=4)
        
    print(f"\nDummy ensemble evaluation complete. Inspect '{ensemble_results_path}' to verify the structure.")

def get_subset_dataloader(data_path: str, img_size: int = 224, batch_size: int = 64, images_per_class: int = 900, num_workers: int = 4) -> DataLoader:
    """
    Creates a DataLoader containing a balanced subset of an ImageFolder dataset.
    """
    # 1. Define transforms dynamically based on img_size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the full dataset
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # 3. Sub-sample images per class
    class_counts = {}
    subset_indices = []

    for idx, (_, class_idx) in enumerate(full_dataset.samples):
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        if class_counts[class_idx] <= images_per_class:
            subset_indices.append(idx)

    print(f"Sub-sampled Dataset from {data_path}: {len(subset_indices)} images total (max {images_per_class} per class).")
    
    # 4. Create the subset and DataLoader
    dataset_subset = Subset(full_dataset, subset_indices)
    loader = DataLoader(
        dataset_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return loader


@torch.no_grad()
def get_model_similarities(model_name, weights_path, support_dir, query_dir, subset_indices, device):
    """
    Extracts the cosine similarity matrix for the query set using a single pretrained model 
    via Prototypical Network logic.
    """
    # 1. Initialize Base Encoder & Image Size
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
        raise ValueError(f"Unknown model: {model_name}")

    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()

    # 2. Define Transforms specific to this model's required resolution
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Setup DataLoaders
    support_dataset = OriginalsOnlyDataset(root=support_dir, transform=transform)
    support_loader = DataLoader(support_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    full_query_dataset = datasets.ImageFolder(root=query_dir, transform=transform)
    query_subset = Subset(full_query_dataset, subset_indices)
    query_loader = DataLoader(query_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Compute Prototypes (L2 Normalized)
    embeddings_list, labels_list = [], []
    for images, labels in tqdm(support_loader, desc=f"[{model_name}] Extracting Support", leave=False):
        images = images.to(device)
        emb = F.normalize(encoder(images), p=2, dim=1)
        embeddings_list.append(emb.cpu())
        labels_list.append(labels.cpu())

    all_embeddings = torch.cat(embeddings_list)
    all_labels = torch.cat(labels_list)
    classes = torch.unique(all_labels)
    
    prototypes = torch.zeros((len(classes), all_embeddings.shape[1])).to(device)
    for i, cls in enumerate(classes):
        cls_mask = (all_labels == cls)
        mean_embed = all_embeddings[cls_mask].mean(dim=0)
        prototypes[i] = F.normalize(mean_embed.unsqueeze(0), p=2, dim=1).squeeze(0)

    # 5. Compute Query Similarities
    all_similarities = []
    all_query_labels = []

    for images, labels in tqdm(query_loader, desc=f"[{model_name}] Evaluating Queries", leave=False):
        images = images.to(device)
        query_embs = F.normalize(encoder(images), p=2, dim=1)
        
        # Cosine similarity matrix: (Batch_Size, Num_Classes)
        sims = torch.matmul(query_embs, prototypes.T)
        
        all_similarities.append(sims.cpu())
        all_query_labels.extend(labels.numpy())

    return torch.cat(all_similarities), np.array(all_query_labels)

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    }

def run_ensembles(experiments_dir, query_dir, weights_dir, results_csv, models_to_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running Ensembles on {device}")

    # 1. Get the deterministic subset indices for the query set (first 900 per class)
    # We just need an empty transform here since we only care about indexing the paths
    temp_dataset = datasets.ImageFolder(root=query_dir, transform=transforms.ToTensor())
    images_per_class = 900
    class_counts = {}
    subset_indices = []
    
    for idx, (_, class_idx) in enumerate(temp_dataset.samples):
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        if class_counts[class_idx] <= images_per_class:
            subset_indices.append(idx)

    # 2. Prepare CSV
    with open(results_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Voting_Type', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])

        dataset_folders = [f for f in Path(experiments_dir).iterdir() if f.is_dir()]
        
        # 3. Iterate through each Few-Shot Experiment
        for dataset_path in dataset_folders:
            dataset_name = dataset_path.name
            print(f"\n{'='*50}\nEnsembling Experiment: {dataset_name}\n{'='*50}")
            
            model_similarities = []
            true_labels = None
            
            # Extract similarity matrices for all available models
            for model_name in models_to_test:
                weights_file = Path(weights_dir) / f"{model_name}_{dataset_name}.pth"
                
                if not weights_file.exists():
                    print(f"⏩ Missing weights for {model_name}, skipping this model in ensemble.")
                    continue
                
                sims, labels = get_model_similarities(
                    model_name=model_name,
                    weights_path=str(weights_file),
                    support_dir=str(dataset_path),
                    query_dir=query_dir,
                    subset_indices=subset_indices,
                    device=device
                )
                
                model_similarities.append(sims)
                if true_labels is None:
                    true_labels = labels # True labels are identical across models
            
            if not model_similarities:
                print(f"No models found for {dataset_name}. Skipping.")
                continue

            # Stack similarities: Shape -> (Num_Models, Num_Queries, Num_Classes)
            stacked_sims = torch.stack(model_similarities)
            
            # --- SOFT VOTING ---
            # 1. Scale similarities by the training temperature (0.07)
            # 2. Apply Softmax along the class dimension (dim=2) to get probabilities
            temperature = 0.07
            probabilities = F.softmax(stacked_sims / temperature, dim=2)
            
            # 3. Sum (or average) the standardized probabilities across models
            soft_sum = torch.sum(probabilities, dim=0) 
            soft_preds = torch.argmax(soft_sum, dim=1).numpy()
            soft_metrics = calculate_metrics(true_labels, soft_preds)
            
            # --- HARD VOTING ---
            # (Hard voting remains unchanged because argmax is invariant to scaling/softmax)
            individual_preds = torch.argmax(stacked_sims, dim=2) 
            hard_preds = torch.mode(individual_preds, dim=0).values.numpy() 
            hard_metrics = calculate_metrics(true_labels, hard_preds)

            print(f"Soft Voting -> Acc: {soft_metrics['accuracy']:.2f}% | F1: {soft_metrics['f1_score']:.2f}%")
            print(f"Hard Voting -> Acc: {hard_metrics['accuracy']:.2f}% | F1: {hard_metrics['f1_score']:.2f}%")

            # Save Results
            writer.writerow([dataset_name, 'Soft', f"{soft_metrics['accuracy']:.2f}", f"{soft_metrics['precision']:.2f}", f"{soft_metrics['recall']:.2f}", f"{soft_metrics['f1_score']:.2f}"])
            writer.writerow([dataset_name, 'Hard', f"{hard_metrics['accuracy']:.2f}", f"{hard_metrics['precision']:.2f}", f"{hard_metrics['recall']:.2f}", f"{hard_metrics['f1_score']:.2f}"])
            f.flush()

if __name__ == "__main__":
    EXPERIMENTS_DIR = "./data/augmented_experiments"
    QUERY_DATA_DIR = '/kaggle/input/cinic10/valid/'
    WEIGHTS_DIR = "./pretrained_encoders_contrastive"
    RESULTS_CSV = "ensemble_evaluation_results.csv"
    MODELS_TO_TEST = ['resnet50', 'efficientnetb4', 'xception']
    
    run_ensembles(EXPERIMENTS_DIR, QUERY_DATA_DIR, WEIGHTS_DIR, RESULTS_CSV, MODELS_TO_TEST)
    print("\n✅ Ensemble Evaluation Complete!")