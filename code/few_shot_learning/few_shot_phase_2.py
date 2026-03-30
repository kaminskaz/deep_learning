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
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    layers_to_unfreeze = []
    
    if model_name == 'efficientnetb4':
        layers_to_unfreeze = list(model.encoder.model.features.children())[-n:]
        
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.projection_head.parameters():
        param.requires_grad = True

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Unfrozen last {n} layers/blocks + projection head. Trainable params: {trainable_count:,}")

class VariableLDataset(datasets.VisionDataset):
    def __init__(self, root, l_max, transform=None):
        super().__init__(root, transform=transform)
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_dir = Path(root) / cls_name
            if not cls_dir.is_dir(): continue
                
            orig_files = list(cls_dir.glob("orig_*"))
            for orig_path in orig_files:
                self.samples.append((str(orig_path), self.class_to_idx[cls_name]))
                
                orig_name = orig_path.name
                aug_files = sorted(list(cls_dir.glob(f"aug_*_{orig_name}")))
                
                for aug_path in aug_files[:l_max]:
                    self.samples.append((str(aug_path), self.class_to_idx[cls_name]))

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
    model_name: str, 
    l_value: int,
    epochs: int = 20, 
    batch_size: int = 64, 
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n--- Training {model_name} | Dataset: {Path(dataset_path).name} | l={l_value} ---")

    base_encoder = EfficientNetB4Encoder()
    embed_dim, img_size = 1792, 380

    model = ProjectionHeadWrapper(base_encoder, embedding_dim=embed_dim).to(device)
    unfreeze_last_n_layers(model, model_name, n=3)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VariableLDataset(root=dataset_path, l_max=l_value, transform=transform)
    print(f"Loaded {len(dataset)} total images (Originals + {l_value} Augs per original).")
    
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
    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_l{l_value}.pth")
    
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    EXPERIMENTS_DIR = Path("./data/augmented_experiments")
    SAVE_DIR = Path("./pretrained_encoders_contrastive")
    SAVE_DIR.mkdir(exist_ok=True)
    
    TARGET_DATASETS = [
        # "10shot_50aug_standard_rotate", 
        # "10shot_50aug_standard_mixed", 
        "10shot_50aug_advanced_mixed"
    ]
    
    L_VALUES = [0, 5, 15, 30, 50]
    
    for dataset_name in TARGET_DATASETS:
        dataset_path = EXPERIMENTS_DIR / dataset_name
        if not dataset_path.exists():
            print(f"Missing dataset: {dataset_path}. Skipping.")
            continue
            
        for l_val in L_VALUES:
            expected_save_path = SAVE_DIR / f"efficientnetb4_{dataset_name}_l{l_val}.pth"
            
            if expected_save_path.exists():
                print(f"Skipping efficientnetb4 on {dataset_name} with l={l_val}: Already trained!")
                continue 
            
            train_supcon(
                dataset_path=str(dataset_path),
                model_name='efficientnetb4',
                l_value=l_val,
                epochs=20, 
                batch_size=32, 
                learning_rate=1e-4
            )