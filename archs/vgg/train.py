import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from vgg import VGG, VGG_CFG
from data.dataset import train_set, valid_set
from cfg import DEVICE, BATCH_SIZE, USE_AMP, EPOCHS, LR, MODEL_CFG


def step(model, optimizer, criterion, loader, step_name='train') -> nn.Module:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    values = {
        'loss': [],
        'predicts': [],
        'targets': []
    }

    with torch.inference_mode(mode=step_name == 'eval'):
        for batch in tqdm(loader, desc=step_name):
            images, targets = [item.to(DEVICE) for item in batch]

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                output = model(images)
                predicts = output.softmax(1)
                loss = criterion(output, targets)
                print(predicts)
                values['loss'].append(loss.item())
                values['predicts'].append(predicts)
                values['targets'].append(targets)
                
                values['loss'].append(loss.item())
                values['predicts'].append(predicts)
                values['targets'].append(targets)
                
                if step_name == 'train':
                    optimizer.zero_grad()
                    
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

            break
        print(values)
    return model


if __name__ == '__main__':
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = VGG(config=MODEL_CFG, num_classes=200).to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for i in range(EPOCHS):
        model.train()
        model = step(model, optimizer, criterion, train_loader)
        
        model.eval()
        step(model, None, criterion, valid_loader, step_name='eval')
                        