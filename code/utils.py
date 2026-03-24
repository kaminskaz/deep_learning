import os
import torch
import torchvision
import numpy as np
import random
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from augmentations.dataset_wrapper import DatasetWrapper
from configs.config import AugmentorConfig

def evaluate(model, data_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    acc = correct / total
    return avg_loss, acc

def get_dataloader(
        directory: str, 
        split: str, 
        batch_size: int,
        preset: Optional[str] = None,
        augmentor_config: Optional[AugmentorConfig] = None,
        seed: int = 42
    ) -> DataLoader:

    dataset = torchvision.datasets.ImageFolder(directory)

    dataset = DatasetWrapper(
        dataset=dataset,
        split=split,
        preset=preset,
        augmentor_config=augmentor_config,
        seed=seed
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=(split=='train'), 
        pin_memory=True,
        num_workers=4,
        persistent_workers=True if split=='train' else False
    )

    return data_loader


def seed_everything(seed: int=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_augmented_samples(
    dataset,
    n_samples:   int   = 8,
    cols:        int   = 4,
    figsize:     tuple = (12, 6),
    denormalize: bool  = True,
):
    """
    Shows n_samples images from a DatasetWrapper after augmentation.

    Args:
        dataset:     A DatasetWrapper instance.
        n_samples:   Number of images to show.
        cols:        Number of columns in the grid.
        figsize:     Figure size.
        denormalize: Whether to reverse normalization for display.
    """
    CINIC_MEAN = np.array([0.47889522, 0.47227842, 0.43047404])
    CINIC_STD  = np.array([0.24205776, 0.23828046, 0.25874835])

    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # grab random indices
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        # tensor → numpy
        img = image.numpy().transpose(1, 2, 0)

        if denormalize:
            img = img * CINIC_STD + CINIC_MEAN

        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(
            f"class: {label}" if not hasattr(dataset.dataset, 'classes')
            else dataset.dataset.classes[label],
            fontsize=9
        )
        axes[i].axis('off')

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f"Augmented samples — split: '{dataset.split}' | "
        f"transform: {dataset.transform.__class__.__name__}",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()


def show_augmentation_effect(
    dataset,
    idx:         int   = 0,
    n_versions:  int   = 8,
    cols:        int   = 4,
    figsize:     tuple = (12, 6),
    denormalize: bool  = True,
):
    """
    Shows the SAME image augmented n_versions times side by side
    so you can see the variation your augmentation produces.
    """
    CINIC_MEAN = np.array([0.47889522, 0.47227842, 0.43047404])
    CINIC_STD  = np.array([0.24205776, 0.23828046, 0.25874835])

    rows = (n_versions + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_versions):
        image, label = dataset[idx]   # each call re-applies random transforms
        img = image.numpy().transpose(1, 2, 0)

        if denormalize:
            img = img * CINIC_STD + CINIC_MEAN

        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"version {i+1}", fontsize=9)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    class_name = (
        dataset.dataset.classes[label]
        if hasattr(dataset.dataset, 'classes')
        else str(label)
    )
    plt.suptitle(
        f"Same image ('{class_name}') — {n_versions} augmented versions",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()