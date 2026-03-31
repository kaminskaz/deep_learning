from torchvision import models
import torch.nn as nn

class EfficientNetB4Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)