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

def main_real():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    seed = 42
    seed_everything(seed)
    
    results_csv_path = "../../results.csv"
    saved_models_dir = "../../saved_models"
    ensemble_results_path = "../../ensemble_results.json"
    num_classes = 10
    
    src_path = kagglehub.dataset_download("mengcius/cinic10")
    data_path = os.path.join(src_path, "valid")

    val_loader = get_subset_dataloader(
        data_path=data_path, 
        img_size=32,
        batch_size=64, 
        images_per_class=900,
        num_workers=4
    )

    print("Parsing best configurations from CSV...")
    best_configs = get_best_models_from_csv(results_csv_path)
    
    model_outputs = {}
    targets = None
    
    for _, row in best_configs.iterrows():
        model_name = row['model_name']
        print(f"Loading best {model_name}...")
        
        try:
            weights_path = find_model_weights(saved_models_dir, row)
        except FileNotFoundError as e:
            print(e)
            continue
            
        model = build_model(model_name, num_classes=num_classes, dropout_rate=row['dropout'])
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        probs, preds, targs = get_model_predictions(model, val_loader, device)
        model_outputs[model_name] = {
            "probs": probs,
            "preds": preds,
            "config": row.to_dict(),
            "weights_path": weights_path
        }
        if targets is None:
            targets = targs 

    # Create combinations: all 3 together, plus pairwise pairs
    model_names = list(model_outputs.keys())
    ensemble_combinations = [model_names] + list(combinations(model_names, 2))
    
    ensemble_records = []

    print("\nEvaluating Ensembles...")
    for combo in ensemble_combinations:
        combo_name = " + ".join(combo)
        print(f"Testing: {combo_name}")
        
        combo_probs = [model_outputs[m]["probs"] for m in combo]
        combo_preds = [model_outputs[m]["preds"] for m in combo]
        
        soft_preds = soft_voting(combo_probs)
        soft_metrics = evaluate_predictions(soft_preds, targets, num_classes)
        
        hard_preds = hard_voting(combo_preds, combo_probs)
        hard_metrics = evaluate_predictions(hard_preds, targets, num_classes)
        
        record = {
            "ensemble_name": combo_name,
            "protocol": "standard_supervised",
            "models_included": [
                {
                    "architecture": m,
                    "csv_val_acc": model_outputs[m]["config"]["val_accuracy"],
                    "weights_path": model_outputs[m]["weights_path"]
                } for m in combo
            ],
            "results": {
                "soft_voting": soft_metrics,
                "hard_voting": hard_metrics
            }
        }
        ensemble_records.append(record)

    with open(ensemble_results_path, "w") as f:
        json.dump(ensemble_records, f, indent=4)
        
    print(f"\nEnsemble evaluation complete. Results saved to {ensemble_results_path}")


if __name__ == "__main__":
    main()