import os
import cv2
import torch
import numpy as np
import albumentations as A
from typing import Any
from torchvision.datasets import ImageFolder
from albumentations.pytorch import ToTensorV2


class Transforms():
    def __init__(self, transforms: A.Compose) -> None:
        self.transforms = transforms

    def __call__(self, image: np.array) -> torch.Tensor:
        return self.transforms(image=image)['image']    


train_transforms = A.Compose([
    A.Resize(224, 224, cv2.INTER_NEAREST, p=1), 
    # A.HorizontalFlip(p=0.35),
    # A.GridDistortion(p=0.15, num_steps=3, distort_limit=0.15,  border_mode=cv2.BORDER_CONSTANT, 
    #                  interpolation=cv2.INTER_CUBIC, value=(255, 255, 255), mask_value=(255, 255, 255)),
    # A.GaussNoise(p=0.2, var_limit=(5, 20)),
    # A.Sharpen(p=0.15, alpha=(0.1, 0.2), lightness=(0.65, 1)),
    A.RGBShift(p=0.5, r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
    A.ColorJitter(p=0.25, brightness=(0.9, 1.3), contrast=(0.9, 1), saturation=(0, 0.2), hue=0),
    A.ChannelShuffle(p=0.01),
    A.CoarseDropout(max_holes=50, max_height=2, max_width=2, p=0.2), 
    # A.ImageCompression(p=0.01, quality_lower=55, quality_upper=90),
    A.ShiftScaleRotate(p=0.15, rotate_limit=5, mask_value=(255, 255, 255), shift_limit_x=0.1, shift_limit_y=0.1, 
                       border_mode=1, scale_limit=0.01, interpolation=3),
    A.Normalize(p=1),
    ToTensorV2(p=1)
])

valid_transforms = A.Compose([
    A.Resize(224, 224, cv2.INTER_NEAREST), 
    A.Normalize(p=1),
    ToTensorV2(p=1)
])
    
train_set = ImageFolder(root=os.path.join(os.path.dirname(__file__), 'tiny-imagenet-200/train'),
                        transform=Transforms(train_transforms),
                        loader=lambda x: cv2.imread(x)[:, :, ::-1])

valid_set = ImageFolder(root=os.path.join(os.path.dirname(__file__), 'tiny-imagenet-200/val'),
                        transform=Transforms(valid_transforms),
                        loader=lambda x: cv2.imread(x)[:, :, ::-1])
