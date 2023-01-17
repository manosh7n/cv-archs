import torch
import torch.nn as nn

from typing import List, Dict, Union


VGG_CFG:  Dict[str, List[Union[str, int]]] = {
    'vgg11': [
        64, 'MaxPool', 128, 'MaxPool', 256, 256, 'MaxPool', 512, 512, 'MaxPool', 512, 512, 'MaxPool'
    ],
    'vgg13': [
        64, 64, 'MaxPool', 128, 128, 'MaxPool', 256, 256, 'MaxPool', 512, 512, 'MaxPool', 512, 512, 'MaxPool'
    ],
    'vgg16': [
        64, 64, 'MaxPool', 128, 128, 'MaxPool', 256, 256, 256, 'MaxPool', 512, 512, 512, 'MaxPool', 512, 512, 512, 'MaxPool'
    ]
}


class VGG(nn.Module):

    def __init__(self,
                 config: str = 'vgg16',
                 num_classes: int = 10,
                 init_weights: bool = False,
                 dropout_p: float = 0.5,
                 pretrained: bool = False
                 ) -> None:
        super().__init__()

        self.init_weights: bool = init_weights
        self.config: List[Union[int, str]] = config
        
        self.feature_extractor = self._make_feature_extractor()
        self.pooling = nn.AdaptiveMaxPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4096, num_classes)
        )
        
        if self.init_weights and not pretrained:
            self._init_weights

        if pretrained:
            pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.pooling(x)
        x = torch.flatten(x)
        x = self.classifier(x)
        
        return x
    
    def _make_feature_extractor(self) -> nn.Sequential:
        in_channels: int = 3
        layers: List[nn.Module] = []
        
        for out in VGG_CFG[self.config]:
            if out == 'MaxPool':
                layers.extend([
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ])
            else:
                layers.extend([
                    nn.Conv2d(in_channels, out, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out),
                    nn.ReLU()
                ])

                in_channels = out
        
        return nn.Sequential(*layers)
    
    def _init_weights(self) -> None:
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                pass
            elif isinstance(module, nn.BatchNorm2d):
                pass
            elif isinstance(module, nn.Conv2d):
                pass
        