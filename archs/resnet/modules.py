import torch
import torch.nn as nn
from typing import Optional


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 shortcut: Optional[nn.Module] = None,
                 start_stage: bool = False,
                 *args, **kwargs
                 ) -> None:
        super().__init__()

        self.downsample = shortcut
        mid_channels = in_channels
        
        if start_stage:
            in_channels *= 2
            self.downsample = ShortcutBlock(in_channels, in_channels // 2, 1)

        self.conv1x1_1 = nn.Conv2d(in_channels, mid_channels, (1, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv3x3 = nn.Conv2d(mid_channels, mid_channels, (3, 3), stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv1x1_2 = nn.Conv2d(mid_channels, out_channels, (1, 1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3x3(out)
        out = self.bn2(out)
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
                 out_channels: int,
                 stride: int,
                 shortcut: Optional[nn.Module] = None,
                 *args, **kwargs
                 ) -> None:
        super().__init__()

        self.downsample = shortcut

        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels, (3, 3), 
                                   stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, (3, 3), 
                                   stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv3x3_1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3x3_2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ShortcutBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 *args, **kwargs
                 ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), 
                              stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
