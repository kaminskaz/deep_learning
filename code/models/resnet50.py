from torchvision import models
import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=False)

        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # change first conv layer to accept 3x32x32 input
        self.model.maxpool = nn.Identity() 
        if dropout_rate > 0:
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                self.model.fc
            )
        else:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)