import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path

from models.efficientnetb4encoder import EfficientNetB4Encoder

from few_shot_phase_1 import SupConLoss, ProjectionHeadWrapper, unfreeze_last_n_layers

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
            
            all_aug_files = sorted(list(cls_dir.glob("aug_*")))
            
            for aug_path in all_aug_files[:l_max]:
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
    l_value : int,
    epochs: int = 20, 
    batch_size: int = 64, 
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"\n--- Training Contrastive Network on: {dataset_path} ---")

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

    dataset = VariableLDataset(root=dataset_path, l_max=l_value, transform=transform)
    
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
    print(f"saved to: {save_path}")

if __name__ == "__main__":
    EXPERIMENTS_DIR = Path("./data/augmented_experiments")
    SAVE_DIR = Path("./pretrained_encoders_contrastive")
    SAVE_DIR.mkdir(exist_ok=True)
    
    TARGET_DATASETS = [
        "10shot_50aug_standard_rotate", 
        "10shot_50aug_standard_mixed", 
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
                print(f"Skipping efficientnetb4 on {dataset_name} with l={l_val}: Already trained")
                continue 
            
            train_supcon(
                dataset_path=str(dataset_path),
                model_name='efficientnetb4',
                l_value=l_val,
                epochs=20, 
                batch_size=32, 
                learning_rate=1e-4
            )