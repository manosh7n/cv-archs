import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from cfg import DEVICE, BATCH_SIZE, USE_AMP
from vgg import VGG, VGG_CFG


def step(model, optimizer, criterion, loader, evaluate=False):
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    with torch.inference_mode(mode=evaluate):
        for batch in loader:
            images, targets = batch.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                output = model(images)
                loss = criterion(output, targets)
                
                if not evaluate:
                    optimizer.zero_grad()
                    
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                        