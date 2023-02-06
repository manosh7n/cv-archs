import torch
import torch.nn as nn

from typing import List, Dict, Union
from modules import BasicResidualBlock, Bottleneck, ShortcutBlock


RESNET_CFG = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3]
}

class ResNet(nn.Module):

    def __init__(self,
                 config: str = '?',
                 num_classes: int = 10,
                 init_weights: bool = False,
                 ) -> None:
        super().__init__()
        
        
        self.input = 64
        self.channels = torch.cumsum(torch.as_tensor([self.input] * 4), dim=0)  # [64, 128, 256, 512]
        
        self.stages = RESNET_CFG[config]
        self.basic_block = Bottleneck if config in ['resnet50', 'resnet101', 'resnet152'] else BasicResidualBlock
        self.increase_factor = 4 if isinstance(self.basic_block, Bottleneck) else 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        self.stage_blocks = nn.Sequential()
        
        for i, stage in enumerate(RESNET_CFG[config]):
            self.stage_blocks.add_module(
                f'stage_block_{i}',
                self._make_block(self.channels[i], stage, i)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def _make_block(self,
                    in_channels: int,
                    block_count: int,
                    block_num: int
                    ) -> nn.Module:
        block = nn.Sequential()
        shortcut = ShortcutBlock(in_channels, 2, self.increase_factor)

        for i in range(1, block_count + 1):
            block.add_module(
                f'basic_block_{block_num}_{i}',
                self.basic_block(in_channels=in_channels,
                                 stride=1,
                                 shortcut=None if i != block_count else shortcut,
                                 increase_factor=1 if i != block_count else self.increase_factor)
            )
            
        return block
        