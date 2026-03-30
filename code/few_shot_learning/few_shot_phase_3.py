import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.efficientnetb4encoder import EfficientNetB4Encoder
from few_shot_phase_1 import SupConLoss, ProjectionHeadWrapper, unfreeze_last_n_layers

class VariableKDataset(datasets.VisionDataset):
    def __init__(self, root, k_max, transform=None):
        super().__init__(root, transform=transform)
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_dir = Path(root) / cls_name
            if not cls_dir.is_dir(): continue
                
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


def train_supcon(
    dataset_path: str, 
    k_value: int,
    epochs: int = 20, 
    batch_size: int = 32, 
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_name = 'efficientnetb4'
    print(f"\n--- Training {model_name} | K={k_value} Shots (l=0) ---")

    # if model_name == 'resnet50':
    #     base_encoder = ResNet50Encoder()
    #     embed_dim, img_size = 2048, 224
    # elif model_name == 'efficientnetb4':
    base_encoder = EfficientNetB4Encoder()
    embed_dim, img_size = 1792, 380
    # elif model_name == 'xception':
    #     base_encoder = XceptionEncoder()
    #     embed_dim, img_size = 2048, 299

    model = ProjectionHeadWrapper(base_encoder, embedding_dim=embed_dim).to(device)
    unfreeze_last_n_layers(model, model_name, n=3)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VariableKDataset(root=dataset_path, k_max=k_value, transform=transform)

    drop_last = len(dataset) > batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last)

    criterion = SupConLoss(temperature=0.07).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            projections = model(images)
            loss = criterion(projections, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    save_dir = "./pretrained_encoders_contrastive"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{model_name}_baseline_k{k_value}.pth")
    
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    EXPERIMENTS_DIR = Path("./data/augmented_experiments")
    MASTER_DATASET_PATH = EXPERIMENTS_DIR / "50shot_0aug_baseline" 
    
    K_VALUES = [5, 10, 15, 20, 30, 40, 50]
    
    for k_val in K_VALUES:
        expected_save_path = Path(f"./pretrained_encoders_contrastive/efficientnetb4_baseline_k{k_val}.pth")
        
        if expected_save_path.exists():
            print(f"Skipping K={k_val}: Already trained")
            continue 
        
        train_supcon(
            dataset_path=str(MASTER_DATASET_PATH),
            k_value=k_val,
            epochs=20, 
            batch_size=32,
            learning_rate=1e-4
        )