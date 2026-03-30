from torchvision import models
import torch.nn as nn

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.model.maxpool = nn.Identity() 
        
        self.model.fc = nn.Identity()
    
    def forward(self, x):
        return self.model(x)