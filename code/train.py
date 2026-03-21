import argparse
import os
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from utils import evaluate
from models.resnet50 import ResNet50
from models.xception import Xception
from models.efficientnetb4 import EfficientNetB4


def train(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    learning_rate: float = 0.02,
    num_epochs: int = 100,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Dict[str, List[float]]]:

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
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

        print(f"Epoch {epoch}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return model, history

def prepare_data_loader(directory: str, batch_size: int, transform: torchvision.transforms.Compose = None) -> DataLoader[Any]:
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    basic_cinic_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=cinic_mean, std=cinic_std)]
    )

    transform = transform if transform is not None else basic_cinic_transform
  
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(directory, transform=transform),
        batch_size=batch_size, 
        shuffle=True
    )

    return data_loader

def main():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10")
    parser.add_argument("--model", choices=["resnet50", "xception", "efficientnetb4"], default="resnet50")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_path", type=str, help="Path to save the trained model")

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    if args.model == "resnet50":
        model = ResNet50(num_classes=10)
    elif args.model == "xception":
        model = Xception(num_classes=10)
    elif args.model == "efficientnetb4":
        model = EfficientNetB4(num_classes=10)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_loader = prepare_data_loader(directory="./data/train", batch_size=args.batch_size)
    val_loader = prepare_data_loader(directory="./data/valid", batch_size=args.batch_size)

    print("Starting training...")

    trained_model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=torch.device(args.device)
    )

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.save(trained_model.state_dict(), f"{args.model_path}/{args.model}_model.pth")

if __name__ == "__main__":
    main()
