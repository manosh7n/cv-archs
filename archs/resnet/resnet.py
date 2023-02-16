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
                 config: str = 'resnet18',
                 num_classes: int = 10,
                 init_weights: bool = False,
                 ) -> None:
        super().__init__()
        
        if config in ['resnet50', 'resnet101', 'resnet152']:
            self.basic_block = Bottleneck
            self.block_inrease_factor = 4
        else:
            self.basic_block = BasicResidualBlock
            self.block_inrease_factor =  2
        
        self.channels = [64, 128, 256, 512]
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                               stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        self.stage_blocks = nn.Sequential()
        for i, stage in enumerate(RESNET_CFG[config]):
            self.stage_blocks.add_module(
                f'stage_block_{i}',
                self._make_stage_block(self.channels[i], stage, i)
            )
        
        # global pool ?
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Linear(self.channels[-1] * self.block_inrease_factor, 
                                    num_classes)
        
        if init_weights:
            self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ?? todo
        for stage_block in self.stage_blocks:
            x = stage_block(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def _make_stage_block(self,
                    in_channels: int,
                    block_count: int,
                    stage_num: int) -> nn.Module:
        
        block = nn.Sequential()
        shortcut = ShortcutBlock(in_channels, stride=2, 
                                 increase_factor=self.block_inrease_factor)
        
        for i in range(block_count - 1):
            block.add_module(
                f'basic_block_{stage_num}_{i}',
                self.basic_block(in_channels=in_channels,
                                 stride=1,
                                 shortcut=None,
                                 increase_factor=1,
                                 first_pos=(i == 0) & (stage_num != 0))
            )
        block.add_module(
            f'basic_block_{stage_num}_{block_count}',
            self.basic_block(in_channels=in_channels,
                             stride=2,
                             shortcut=shortcut,
                             increase_factor=self.block_inrease_factor)
            )
        print(self.block_inrease_factor, in_channels)
            
        return block
    
    def _init_weights(self) -> None:
        pass
        