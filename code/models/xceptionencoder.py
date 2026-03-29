import timm
import torch.nn as nn

class XceptionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model('xception', pretrained=True, num_classes=0)

    def forward(self, x):
        return self.model(x)