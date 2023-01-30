import torch
import torch.nn as nn

from typing import List, Dict, Union
from .modules import *


class ResNet(nn.Module):

    def __init__(self,
                 config: str = '?',
                 num_classes: int = 10,
                 init_weights: bool = False,
                 ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass