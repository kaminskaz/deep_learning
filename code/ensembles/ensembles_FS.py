import os
import csv
import json
import sys
import kagglehub
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join("../", os.getcwd(), "code")))
# Custom module imports from your environment
from models.resnet50encoder import ResNet50Encoder
from models.efficientnetb4encoder import EfficientNetB4Encoder
from models.xceptionencoder import XceptionEncoder
from contrastive_evaluation_1 import OriginalsOnlyDataset 


def get_best_models_from_csv(csv_path: str, top_k: int = 3) -> pd.DataFrame:
    """
    Reads the results CSV and selects the best configuration for each architecture
    based on validation accuracy.
    """
    df = pd.read_csv(csv_path)
    
    if 'phase2' in csv_path:
        idx = df.groupby(['Experiment', 'L_Value'])['Accuracy'].idxmax()
    else:
        idx = df.groupby(['Experiment', 'Model'])['Accuracy'].idxmax()
    best_models = df.loc[idx]
    best_models = best_models.head(top_k).reset_index(drop=True)
    return best_models

def find_model_weights(saved_models_dir: str, row: pd.Series) -> str:
    """
    Reconstructs the expected filename: {model}_{experiment}_l{L}.pth or {model}_{experiment}.pth
    """
    model_name = row['Model']
    experiment = row['Experiment']
    
    l_val = row.get('L_Value')
    
    if pd.notna(l_val):
        filename = f"{model_name}_{experiment}_l{int(l_val)}.pth"
    else:
        filename = f"{model_name}_{experiment}.pth"

    weights_file = Path(saved_models_dir) / filename

    if weights_file.exists():
        return str(weights_file)
    
    raise FileNotFoundError(f"Missing weights at: {weights_file}")

def test_get_best_models():
    models_dir = "pretrained_encoders_contrastive"
    csv_path1 = "evaluation_results.csv"
    csv_path2 = "evaluation_results_phase2.csv"

    df1 = get_best_models_from_csv(csv_path=csv_path1, top_k=1)
    df2 = get_best_models_from_csv(csv_path=csv_path2, top_k=1)

    print("--- Phase 1 Best ---")
    print(df1)
    for _, row in df1.iterrows():
        try:
            print(f"Weights found: {find_model_weights(models_dir, row)}")
        except FileNotFoundError as e:
            print(e)

    print("\n--- Phase 2 Best ---")
    print(df2)
    for _, row in df2.iterrows():
        try:
            print(f"Weights found: {find_model_weights(models_dir, row)}")
        except FileNotFoundError as e:
            print(e)

def get_query_subset_indices(query_dir: str, images_per_class: int = 900) -> list:
    """
    Extracts deterministic subset indices for the query dataset.
    """
    temp_dataset = datasets.ImageFolder(root=query_dir, transform=transforms.ToTensor())
    class_counts = {}
    subset_indices = []
    
    for idx, (_, class_idx) in enumerate(temp_dataset.samples):
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        if class_counts[class_idx] <= images_per_class:
            subset_indices.append(idx)
            
    return subset_indices

@torch.no_grad()
def get_model_similarities(model_name: str, weights_path: str, support_dir: str, query_dir: str, subset_indices: list, device: torch.device):
    """
    Extracts the cosine similarity matrix for the query set using a single pretrained model 
    via Prototypical Network logic.
    """
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

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    support_dataset = OriginalsOnlyDataset(root=support_dir, transform=transform)
    support_loader = DataLoader(support_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    full_query_dataset = datasets.ImageFolder(root=query_dir, transform=transform)
    query_subset = Subset(full_query_dataset, subset_indices)
    query_loader = DataLoader(query_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

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

    all_similarities = []
    all_query_labels = []

    for images, labels in tqdm(query_loader, desc=f"[{model_name}] Evaluating Queries", leave=False):
        images = images.to(device)
        query_embs = F.normalize(encoder(images), p=2, dim=1)
        sims = torch.matmul(query_embs, prototypes.T)
        
        all_similarities.append(sims.cpu())
        all_query_labels.extend(labels.numpy())

    return torch.cat(all_similarities), np.array(all_query_labels)


def soft_voting(combo_probs: list, temperature: float = 0.07) -> np.ndarray:
    """
    Applies Softmax scaling using a temperature parameter, averages the 
    probabilities across all models in the combo, and returns the argmax predictions.
    """
    stacked_sims = torch.stack(combo_probs)
    probabilities = F.softmax(stacked_sims / temperature, dim=2)
    soft_sum = torch.sum(probabilities, dim=0) 
    return torch.argmax(soft_sum, dim=1).numpy()


def hard_voting(combo_preds: list) -> np.ndarray:
    """
    Calculates individual model predictions and performs majority class voting.
    """
    stacked_preds = torch.stack(combo_preds)
    return torch.mode(stacked_preds, dim=0).values.numpy()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates macro-averaged metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    }


def run_ensembles(experiments_dir: str, query_dir: str, weights_dir: str, hyperparams_csv: str, results_json_path: str):
    """
    Main execution pipeline for evaluating combinations of ensembles and saving to JSON.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Running Ensembles on {device}")

    # 1. Get query subsets and parse best configs
    subset_indices = get_query_subset_indices(query_dir, images_per_class=900)
    print("Parsing best configurations from CSV...")
    best_configs = get_best_models_from_csv(hyperparams_csv)
    
    dataset_folders = [f for f in Path(experiments_dir).iterdir() if f.is_dir()]
    ensemble_records = []

    # 2. Iterate through each Few-Shot Experiment
    for dataset_path in dataset_folders:
        dataset_name = dataset_path.name
        print(f"\n{'='*50}\nEnsembling Experiment: {dataset_name}\n{'='*50}")
        
        model_outputs = {}
        targets = None
        
        # Extract predictions for all models
        for _, row in best_configs.iterrows():
            model_name = row['model_name']
            print(f"Loading best {model_name}...")
            
            try:
                weights_path = find_model_weights(weights_dir, row, dataset_name)
            except FileNotFoundError as e:
                print(e)
                continue
                
            sims, labels = get_model_similarities(
                model_name=model_name,
                weights_path=weights_path,
                support_dir=str(dataset_path),
                query_dir=query_dir,
                subset_indices=subset_indices,
                device=device
            )
            
            # For few-shot, 'probs' are the similarities, 'preds' are the argmax of those similarities
            model_outputs[model_name] = {
                "probs": sims,
                "preds": torch.argmax(sims, dim=1),
                "config": row.to_dict(),
                "weights_path": weights_path
            }
            if targets is None:
                targets = labels 
        
        if not model_outputs:
            print(f"No models successfully loaded for {dataset_name}. Skipping combinations.")
            continue

        model_names = list(model_outputs.keys())
        ensemble_combinations = [model_names] + list(combinations(model_names, 2)) if len(model_names) > 1 else [model_names]

        print(f"\nEvaluating Combinations for {dataset_name}...")
        for combo in ensemble_combinations:
            combo_name = " + ".join(combo)
            print(f"Testing: {combo_name}")
            
            combo_probs = [model_outputs[m]["probs"] for m in combo]
            combo_preds = [model_outputs[m]["preds"] for m in combo]
            
            soft_preds = soft_voting(combo_probs, temperature=0.07)
            soft_metrics = evaluate_predictions(targets, soft_preds)
            
            hard_preds = hard_voting(combo_preds)
            hard_metrics = evaluate_predictions(targets, hard_preds)
            
            record = {
                "experiment_dataset": dataset_name,
                "ensemble_name": combo_name,
                "protocol": "few_shot_prototypical",
                "models_included": [
                    {
                        "architecture": m,
                        "csv_val_acc": model_outputs[m]["config"].get("val_accuracy", "N/A"),
                        "weights_path": model_outputs[m]["weights_path"]
                    } for m in combo
                ],
                "results": {
                    "soft_voting": soft_metrics,
                    "hard_voting": hard_metrics
                }
            }
            ensemble_records.append(record)

    # 5. Save to JSON
    with open(results_json_path, "w") as f:
        json.dump(ensemble_records, f, indent=4)
        
    print(f"\n✅ Ensemble evaluation complete. Results saved to {results_json_path}")


if __name__ == "__main__":
    EXPERIMENTS_DIR = "data/augmented_experiments"
    src_path = kagglehub.dataset_download("mengcius/cinic10")
    QUERY_DATA_DIR = os.path.join(src_path, "valid")
    WEIGHTS_DIR = "pretrained_encoders_contrastive" # The files expected by find_model_weights
    HYPERPARAMS_CSV = "evaluation_results.csv" # The file parsed by get_best_models_from_csv
    RESULTS_JSON = "ensemble_evaluation_results_contrastive.json"
    
    run_ensembles(EXPERIMENTS_DIR, QUERY_DATA_DIR, WEIGHTS_DIR, HYPERPARAMS_CSV, RESULTS_JSON)