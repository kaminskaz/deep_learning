import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import gc

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.resnet50encoder import ResNet50Encoder
from models.efficientnetb4encoder import EfficientNetB4Encoder
from models.xceptionencoder import XceptionEncoder

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    """
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

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(batch_size).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - mean_log_prob_pos.mean()
        return loss

class ProjectionHeadWrapper(nn.Module):
    def __init__(self, encoder, embedding_dim, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        projections = self.projection_head(embeddings)
        return projections

def unfreeze_last_n_layers(model, model_name, n=3):
    """
    Freezes the entire backbone, then unfreezes the last n logical blocks.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False

    layers_to_unfreeze = []
    
    if model_name == 'resnet50':
        layers_to_unfreeze = list(model.encoder.model.layer4.children())[-n:]
        
    elif model_name == 'efficientnetb4':
        layers_to_unfreeze = list(model.encoder.model.features.children())[-n:]
        
    elif model_name == 'xception':
        for i in range(12 - n + 1, 13): 
            block_name = f'block{i}'
            if hasattr(model.encoder.model, block_name):
                layers_to_unfreeze.append(getattr(model.encoder.model, block_name))
        
        for exit_layer in ['conv3', 'bn3', 'conv4', 'bn4']:
            if hasattr(model.encoder.model, exit_layer):
                layers_to_unfreeze.append(getattr(model.encoder.model, exit_layer))
        
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.projection_head.parameters():
        param.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Unfrozen last {n} layers/blocks + projection head. Trainable params: {trainable_count:,}")

def train_supcon(
    dataset_path: str, 
    model_name: str, 
    epochs: int = 20, 
    batch_size: int = 64, 
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"\n--- Training Contrastive Network on: {dataset_path} ---")

    if model_name == 'resnet50':
        base_encoder = ResNet50Encoder()
        embed_dim, img_size = 2048, 224
    elif model_name == 'efficientnetb4':
        base_encoder = EfficientNetB4Encoder()
        embed_dim, img_size = 1792, 380
    elif model_name == 'xception':
        base_encoder = XceptionEncoder()
        embed_dim, img_size = 2048, 299

    model = ProjectionHeadWrapper(base_encoder, embedding_dim=embed_dim).to(device)

    unfreeze_last_n_layers(model, model_name, n=3)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

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
    dataset_name = os.path.basename(dataset_path)
    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}.pth")
    
    torch.save(model.encoder.state_dict(), save_path)
    print(f"saved to: {save_path}")

if __name__ == "__main__":
    EXPERIMENTS_DIR = Path("./data/augmented_experiments")
    MODELS_TO_TEST = ['resnet50', 'efficientnetb4', 'xception']
    SAVE_DIR = Path("./pretrained_encoders_contrastive")
    SAVE_DIR.mkdir(exist_ok=True) 

    dataset_folders = [f for f in EXPERIMENTS_DIR.iterdir() if f.is_dir()]
    
    if not dataset_folders:
        print(f"No datasets found ")
    else:
        print(f"Found {len(dataset_folders)} datasets. Starting master training loop...")
        
        for dataset_path in dataset_folders:
            dataset_name = dataset_path.name 
            
            for model_name in MODELS_TO_TEST:
                expected_save_path = SAVE_DIR / f"{model_name}_{dataset_name}.pth"
                
                if expected_save_path.exists():
                    print(f"Skipping {model_name} on {dataset_name}: Already trained")
                    continue
                
                train_supcon(
                    dataset_path=str(dataset_path),
                    model_name=model_name,
                    epochs=20, 
                    batch_size=32, 
                    learning_rate=1e-4
                )
        