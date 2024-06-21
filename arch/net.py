#!/usr/bin/env python3

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F

#--->>> Convolutional NN

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.25)
        x = F.leaky_relu(self.fc1(x.view(x.shape[0], -1)))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

#--->>> ResNet

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_chan: int,
                 out_chan: int,
                 stride: int=1,
                 downsample: nn.Module = None
                 ) -> None:
        
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chan, out_chan*self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan * self.expansion)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,
                 img_channels: int,
                 num_layers: int,
                 block: Type[BasicBlock],
                 num_classes: int=10) -> None:
        
        super(ResNet, self).__init__()
        layers = self._get_layers(num_layers)
        self.expansion = block.expansion

        self.in_chan = 64

        self.conv1 = nn.Conv2d(in_channels = img_channels,
                               out_channels=self.in_chan,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.in_chan)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _get_layers(self, num_layers):
        if num_layers == 18:
            return [2, 2, 2, 2]
        elif num_layers == 34:
            return [3, 4, 6, 3]
        elif num_layers == 50:
            return [3, 4, 6, 3]
        elif num_layers == 101:
            return [3, 4, 23, 3]
        elif num_layers == 152:
            return [3, 8, 36, 3]
        else:
            raise ValueError("Unsupported ResNet model")

    def _make_layer(self,
                    block: Type[BasicBlock],
                    out_chan: int,
                    blocks: int,
                    stride: int=1) -> nn.Sequential:
        
        downsample = None
        if stride != 1 or self.in_chan != out_chan * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_chan, out_chan*block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_chan*block.expansion))

        layers = []
        layers.append(block(self.in_chan, out_chan, stride, downsample))
        self.in_chan = out_chan * block.expansion

        for _ in range (1, blocks):
            layers.append(block(self.in_chan, out_chan))

        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x