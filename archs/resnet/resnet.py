import torch
import torch.nn as nn

from typing import List, Dict, Union


class ResNet(nn.Module):

    def __init__(self,
                 config: str = '?',
                 num_classes: int = 10,
                 init_weights: bool = False,
                 ) -> None:
        super().__init__()
        