import os
import sys
import pandas as pd
import torch
from torchmetrics import Precision, Recall, F1Score, Accuracy
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import get_dataloader, seed_everything
from models.resnet50 import ResNet50
from models.xception import Xception
from models.efficientnetb4 import EfficientNetB4

def build_model(model_name: str, num_classes: int, dropout_rate: float = 0.0) -> torch.nn.Module: 
    models = {
        'resnet50': ResNet50,   
        'xception': Xception,
        'efficientnetb4': EfficientNetB4,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")
    return models[model_name](num_classes=num_classes, dropout_rate=dropout_rate)

def calculate_metrics(model, data_loader, num_classes, device='cpu'):
    model.eval()
    model.to(device)
    all_preds = []
    all_targets = []

    precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Processing batches", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    precision_value = precision(all_preds, all_targets).item()
    recall_value = recall(all_preds, all_targets).item()
    f1_value = f1_score(all_preds, all_targets).item()
    accuracy_value = accuracy(all_preds, all_targets).item()

    return precision_value, recall_value, f1_value, accuracy_value


def main():
    seed = 42
    batch_size = 64
    seed_everything(seed)
    saved_models_dir = "saved_models"
    results_csv_path = "results.csv"

    print("Dataloader preparation...")
    # ── dataloader ────────────────────────────────────────────
    val_loader = get_dataloader(
        directory="data/valid",
        split='val',
        batch_size=batch_size,
        seed=seed
    )

    print("Dataloader ready. Processing saved models...")
    csv_rows = []

    for model_dir in os.listdir(saved_models_dir):
        model_path = os.path.join(saved_models_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        for file in os.listdir(model_path):
            if "drop" in file:
                continue

            if file.endswith(".pth"):
                print(f"Processing {file}...")
                full_path = os.path.join(model_path, file)
                
                # ex. efficientnetb4_experiment_1_efficientnet4b_bs32_adam_lr0.001.pth
                name_parts = file[:-4].split("_")  # Remove .pth and split by "_"
                
                # Extract parameters (customize depending on your naming convention)
                try:
                    model_name = name_parts[0]  
                    experiment_num = int(name_parts[2] )
                    batch_size = int(name_parts[4].replace("bs", ""))
                    optimizer = name_parts[5]
                    lr = float(name_parts[6].replace("lr", ""))
                    dropout = 0.0
                    regularization = "none"
                    lambda_value = 0.0

                except IndexError:
                    print(f"Filename {file} doesn't match expected pattern. Skipping...")
                    continue

                model = build_model(model_name, num_classes=10, dropout_rate=dropout)
                checkpoint = torch.load(full_path, map_location='cpu')
                model_dict = checkpoint['model']
                history = checkpoint['history']

                model.load_state_dict(model_dict)

                print("Calculating metrics on validation set...")
                precision, recall, f1, accuracy = calculate_metrics(model, val_loader, num_classes=10, device='mps')

                num_epochs = len(history['train_loss'])

                # Save row for CSV
                row = {
                    "model_name": model_name,
                    "experiment": experiment_num,
                    "batch_size": batch_size,
                    "optimizer": optimizer,
                    "lr": lr,
                    "dropout": dropout,
                    "regularization": regularization,
                    "lambda": lambda_value,
                    "history": history,
                    "num_epochs": num_epochs,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1_score": f1,
                    "val_accuracy": accuracy
                }
                
                df = pd.DataFrame([row])

                df.to_csv(
                    results_csv_path,
                    mode='a',              # append
                    header=not os.path.exists(results_csv_path),  # write header only once
                    index=False
                )

                new_name = f"{file[:-4]}_reg{regularization}_lambda{lambda_value}_drop{dropout}.pth"
                new_full_path = os.path.join(model_path, new_name)
                
                os.rename(full_path, new_full_path)
                print(f"Renamed to {new_name}")


if __name__ == "__main__":
    main()