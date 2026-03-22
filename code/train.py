import argparse
import copy
import os
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import random

from utils import evaluate, get_dataloader, seed_everything, show_augmentation_effect, show_augmented_samples
from models.resnet50 import ResNet50
from models.xception import Xception
from models.efficientnetb4 import EfficientNetB4
from augmentations.dataset_wrapper import DatasetWrapper
from augmentations.augmentor import Augmentor
from configs.config import TrainingConfig, load_config, AugmentorConfig, load_augmentor_config


def train(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    num_epochs: int = 30,
    patience: int = 5,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, List[float]]]:

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None

    model = model.to(device)

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(dim=1)
            correct += (predicted == targets.to(device)).sum().item()
            total += len(predicted)

            loop.set_postfix(
                loss=f"{running_loss / (total / inputs.size(0)):.4f}",
            )
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step() 

        print(f"Epoch {epoch}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if (epoch) % 5 == 0:
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':             val_loss,
                    'history':              history
                }, f"checkpoint_epoch{epoch}.pth")
                print(f"Checkpoint saved at epoch {epoch}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_weights)
                torch.save(best_weights, f"best_model.pth")
                break
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return model, history


def build_model(cfg: TrainingConfig) -> nn.Module:
    models = {
        'resnet50': ResNet50,
        'xception': Xception,
        'efficientnetb4': EfficientNetB4,
    }
    if cfg.model not in models:
        raise ValueError(f"Unknown model '{cfg.model}'. Choose from: {list(models.keys())}")
    return models[cfg.model](num_classes=cfg.num_classes, dropout_rate=cfg.dropout_rate)


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    optimizers = {
        'sgd':   lambda: torch.optim.SGD(
                     model.parameters(),
                     lr=cfg.learning_rate,
                     momentum=cfg.momentum
                 ),
        'adam':  lambda: torch.optim.Adam(
                     model.parameters(),
                     lr=cfg.learning_rate
                 )}
    
    if cfg.optimizer not in optimizers:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Choose from: {list(optimizers.keys())}")
    return optimizers[cfg.optimizer]()


def main():
    show_augmentation_preview = False 
    parser = argparse.ArgumentParser(description="Train a model on CINIC-10")
    parser.add_argument('--config', type=str, default='code/configs/default.yaml',
                        help='Path to training config YAML file')
    parser.add_argument('--version', type=str, default='v1',)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    print(f"Loaded config: {args.config}")
    print(f"Device: {cfg.device} | Model: {cfg.model} | Epochs: {cfg.num_epochs}")

    # ── augmentor ──────────────────────────────────────────────
    augmentor_config = None
    if cfg.augmentor_config:
        augmentor_config = load_augmentor_config(cfg.augmentor_config)

    # ── dataloaders ────────────────────────────────────────────
    train_loader = get_dataloader(
        directory=cfg.train_dir,
        split='train',
        batch_size=cfg.batch_size,
        preset=cfg.preset,
        augmentor_config=augmentor_config,
        seed=cfg.seed
    )
    val_loader = get_dataloader(
        directory=cfg.val_dir,
        split='val',
        batch_size=cfg.batch_size,
        preset=cfg.preset,
        augmentor_config=augmentor_config,
        seed=cfg.seed
    )

    # ── preview augmentation ───────────────────────────────────
    if show_augmentation_preview:
        print("Previewing augmentation...")
        show_augmented_samples(train_loader.dataset, n_samples=3)
        show_augmentation_effect(train_loader.dataset, idx=0, n_versions=3)

    # ── model, optimizer, scheduler ───────────────────────────
    model = build_model(cfg).to(cfg.device)
    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    # ── train ──────────────────────────────────────────────────
    print("Starting training...")
    trained_model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=cfg.num_epochs,
        patience=cfg.patience,
        device=torch.device(cfg.device),
    )

    # ── save ───────────────────────────────────────────────────
    dirname = f"{cfg.model_path}/{cfg.model}/"
    os.makedirs(dirname, exist_ok=True)
    save_path = f"{dirname}/{cfg.model}_{args.version}.pth"
    torch.save({'model': trained_model.state_dict(), 'history': history}, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()