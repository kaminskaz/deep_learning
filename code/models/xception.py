from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init


class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False
        ):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            reps, 
            stride=1, 
            start_with_relu=True, 
            grow_first=True
        ):
        super(Block, self).__init__()

        super().__init__()

        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []

        for i in range(reps):
            if grow_first:
                in_ch  = in_channels  if i == 0         else out_channels
                out_ch = out_channels if i == 0         else out_channels
            else:
                in_ch  = in_channels  if i < reps - 1   else in_channels
                out_ch = in_channels  if i < reps - 1   else out_channels

            if start_with_relu:
                layers.append(nn.ReLU())

            layers.append(SeparableConv2d(in_ch, out_ch))
            layers.append(nn.BatchNorm2d(out_ch))

        if stride > 1:
            layers.append(nn.MaxPool2d(3, stride=stride, padding=1))

        self.layers = nn.Sequential(*layers)


    def forward(self, x):
            residual = self.skip(x) if self.skip else x
            return self.layers(x) + residual



class Xception(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # ------- entry flow --------
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3 , stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.block1=Block(64, 128, reps=2, stride=2, start_with_relu=False, grow_first=True)
        self.block2=Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3=Block(256, 728, reps=2, stride=2, start_with_relu=True, grow_first=True)
        
        # ------- middle flow --------
        self.middle = nn.Sequential(*[
            Block(728, 728, reps=3, stride=1, start_with_relu=True, grow_first=True)
            for _ in range(8)
        ])

        # ------- exit flow --------
        self.block12=Block(728, 1024, reps=2, stride=1, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU()

        self.conv4 = SeparableConv2d(1536, 2048)
        self.bn4 = nn.BatchNorm2d(2048)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()           
        )

        self.fc = nn.Linear(2048, num_classes)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
            # ------- entry flow --------
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)

            # ------- middle flow --------
            x = self.middle(x)

            # ------ exit flow --------
            x = self.block12(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))

            x = self.global_pool(x)
            x = self.dropout(x)
            x = self.fc(x)

            return x

