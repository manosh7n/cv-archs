import torch
import torch.nn as nn
from typing import Optional


class Bottleneck(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 stride: int, 
                 downsample_block: Optional[nn.Module] = None,
                 increase_factor: int = 4) -> None:
        super().__init__()
        
        self.increase_factor = increase_factor
        self.downsample = downsample_block
        
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, (3, 3), stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        self.conv1x1_2 = nn.Conv2d(in_channels, in_channels * self.increase_factor, (1, 1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(in_channels * self.increase_factor)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1x1_1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv3x3(out)
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.conv1x1_2(out)
        out = self.bn3(out)
        
        if self.downsample:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out
        

class BasicResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int, 
                 stride: int, 
                 downsample_block: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.downsample = downsample_block
        
        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels, (3, 3), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.conv3x3_2 = nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv3x3_1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv3x3_2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out
        
        
        
        
        