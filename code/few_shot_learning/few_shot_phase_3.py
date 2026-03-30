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


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        return - mean_log_prob_pos.mean()

class ProjectionHeadWrapper(nn.Module):
    def __init__(self, encoder, embedding_dim, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(nn.Linear(embedding_dim, projection_dim))
    def forward(self, x):
        return self.projection_head(self.encoder(x))

def unfreeze_last_n_layers(model, model_name, n=3):
    for param in model.encoder.parameters(): param.requires_grad = False
    layers_to_unfreeze = list(model.encoder.model.features.children())[-n:] if model_name == 'efficientnetb4' else []
    for layer in layers_to_unfreeze:
        for param in layer.parameters(): param.requires_grad = True
    for param in model.projection_head.parameters(): param.requires_grad = True

class VariableKDataset(datasets.VisionDataset):
    """
    Loads strictly the first 'k_max' original images per class from a target directory.
    """
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


def train_supcon_k_experiment(
    dataset_path: str, 
    k_value: int,
    epochs: int = 20, 
    batch_size: int = 32, 
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_name = 'efficientnetb4'
    print(f"\n--- Training {model_name} | K={k_value} Shots (l=0) ---")

    base_encoder = EfficientNetB4Encoder()
    embed_dim, img_size = 1792, 380

    model = ProjectionHeadWrapper(base_encoder, embedding_dim=embed_dim).to(device)
    unfreeze_last_n_layers(model, model_name, n=3)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VariableKDataset(root=dataset_path, k_max=k_value, transform=transform)
    print(f"Loaded {len(dataset)} total images ({k_value} per class).")

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
    
    if not MASTER_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing master dataset at {MASTER_DATASET_PATH}. Run generation script first.")
    
    K_VALUES = [5, 10, 15, 20, 30, 40, 50]
    
    for k_val in K_VALUES:
        expected_save_path = Path(f"./pretrained_encoders_contrastive/efficientnetb4_baseline_k{k_val}.pth")
        
        if expected_save_path.exists():
            print(f"Skipping K={k_val}: Already trained!")
            continue 
        
        train_supcon_k_experiment(
            dataset_path=str(MASTER_DATASET_PATH),
            k_value=k_val,
            epochs=20, 
            batch_size=32,
            learning_rate=1e-4
        )