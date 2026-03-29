import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import gc

# Assuming these return the models with the classification heads set to nn.Identity()
from models.resnet50encoder import ResNet50Encoder
from models.efficientnetb4encoder import EfficientNetB4Encoder
from models.xceptionencoder import XceptionEncoder

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Uses Cosine Similarity (via L2 norm + dot product) to pull same classes together.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # L2 Normalize features to project them onto a unit sphere
        features = F.normalize(features, p=2, dim=1)
        
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        # Mask of positive pairs (same class)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # Compute Cosine Similarity matrix (dot product of L2 normalized vectors)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast (the diagonal)
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
    """
    Wraps the encoder with an MLP projection head for contrastive pretraining.
    """
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
    # 1. Freeze the underlying encoder first
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # 2. Target specific architecture ends
    layers_to_unfreeze = []
    
    if model_name == 'resnet50':
        # ResNet neatly groups the final layers in layer4
        layers_to_unfreeze = list(model.encoder.model.layer4.children())[-n:]
        
    elif model_name == 'efficientnetb4':
        # EfficientNet neatly groups them in features
        layers_to_unfreeze = list(model.encoder.model.features.children())[-n:]
        
    elif model_name == 'xception':
        # timm's Xception registers blocks individually (block1... block12)
        # If n=3, this loop grabs block10, block11, block12
        for i in range(12 - n + 1, 13): 
            block_name = f'block{i}'
            if hasattr(model.encoder.model, block_name):
                layers_to_unfreeze.append(getattr(model.encoder.model, block_name))
        
        # We also need to unfreeze the final "exit flow" convolutions 
        # that sit between block12 and the classification head.
        for exit_layer in ['conv3', 'bn3', 'conv4', 'bn4']:
            if hasattr(model.encoder.model, exit_layer):
                layers_to_unfreeze.append(getattr(model.encoder.model, exit_layer))
        
    # 3. Unfreeze the targeted backbone layers
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    # 4. ALWAYS unfreeze the projection head
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

    # 1. Initialize Encoder & Scaling Size
    if model_name == 'resnet50':
        base_encoder = ResNet50Encoder()
        embed_dim, img_size = 2048, 224
    elif model_name == 'efficientnetb4':
        base_encoder = EfficientNetB4Encoder()
        embed_dim, img_size = 1792, 380
    elif model_name == 'xception':
        base_encoder = XceptionEncoder()
        embed_dim, img_size = 2048, 299
    else:
        raise ValueError("Invalid model name.")

    # Wrap encoder in the projection head and move to device
    model = ProjectionHeadWrapper(base_encoder, embedding_dim=embed_dim).to(device)

    # 2. Freeze all but the last 3 layers + projection head
    unfreeze_last_n_layers(model, model_name, n=3)
    
    # 3. Dynamic Transform for Model Scaling
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Standard DataLoader (Contrastive learning uses standard batches, not episodes)
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # 5. Loss & Optimizer (Only passing parameters that require gradients)
    criterion = SupConLoss(temperature=0.07).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 6. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass: get projected embeddings
            projections = model(images)
            
            # Calculate SupCon Loss (Cosine Similarity logic is inside)
            loss = criterion(projections, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    # 7. Save ONLY the underlying encoder (throw away the projection head)
    save_dir = "./pretrained_encoders_contrastive"
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = os.path.basename(dataset_path)
    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}.pth")
    
    # Notice we save model.encoder.state_dict(), leaving the projection head behind
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Pretraining complete! Encoder saved to: {save_path}")

# --- Example Execution for Phase 1 ---
if __name__ == "__main__":
    EXPERIMENTS_DIR = Path("./data/augmented_experiments")
    MODELS_TO_TEST = ['resnet50', 'efficientnetb4', 'xception']
    
    # Define where the models are going to be saved
    SAVE_DIR = Path("./pretrained_encoders_contrastive")
    SAVE_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist yet
    
    dataset_folders = [f for f in EXPERIMENTS_DIR.iterdir() if f.is_dir()]
    
    if not dataset_folders:
        print(f"No datasets found in {EXPERIMENTS_DIR}.")
    else:
        print(f"Found {len(dataset_folders)} datasets. Starting master training loop...")
        
        for dataset_path in dataset_folders:
            dataset_name = dataset_path.name # e.g., "10shot_50aug_standard_blur"
            
            for model_name in MODELS_TO_TEST:
                # 1. Construct the exact path where this model WOULD be saved
                expected_save_path = SAVE_DIR / f"{model_name}_{dataset_name}.pth"
                
                # 2. Check if it already exists
                if expected_save_path.exists():
                    print(f"⏩ Skipping {model_name} on {dataset_name}: Already trained!")
                    continue # Skip to the next model
                
                # 3. If it doesn't exist, proceed with training
                train_supcon(
                    dataset_path=str(dataset_path),
                    model_name=model_name,
                    epochs=20, 
                    batch_size=32, 
                    learning_rate=1e-4
                )
        
        print("\nAll Pretraining Complete!")