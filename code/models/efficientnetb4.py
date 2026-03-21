from torchvision import models
import torch.nn as nn
import torch

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.model = models.efficientnet_b4(pretrained=False)


        self.model.features[0][0] = nn.Conv2d(
            in_channels=3,
            out_channels=48,        
            kernel_size=3,
            stride=1,             
            padding=1,
            bias=False
        )

        in_features = self.model.classifier[1].in_features
        
        if dropout_rate > 0:
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)