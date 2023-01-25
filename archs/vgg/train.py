import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import wandb
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, accuracy, recall, precision, f1_score

from vgg import VGG
from data.dataset import train_set, valid_set
from cfg import DEVICE, BATCH_SIZE, USE_AMP, EPOCHS, LR, MODEL_CFG


run = wandb.init(project="vgg_classifier")
artifact = wandb.Artifact('model', type='model')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def step(model, optimizer, criterion, loader, epoch, step_name='train') -> nn.Module:
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
                predicts = output.softmax(1).argmax(1)
                loss = criterion(output, targets)
   
                values['loss'].append(loss.item())
                values['predicts'].append(output)
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
        values['predicts'] = torch.cat(values['predicts'], dim=0)
        values['targets'] = torch.cat(values['targets'], dim=0)
        
        metric_params = dict(
            preds=values['predicts'],
            target=values['targets'],
            task='multiclass', 
            num_classes=200
        )
        print(metric_params['preds'].shape, metric_params['target'].shape)
        run.log(
            {
                f'{step_name}/acc': accuracy(**metric_params),
                f'{step_name}/f1_micro': f1_score(average='micro', **metric_params),
                f'{step_name}/f1_macro': f1_score(average='macro', **metric_params),
                f'{step_name}/auroc_macro': auroc(average='macro', **metric_params),
                f'{step_name}/precision_micro': precision(average='micro', **metric_params),
                f'{step_name}/precision_macro': precision(average='macro', **metric_params),
                f'{step_name}/recall_micro': recall(average='macro', **metric_params),
                f'{step_name}/recall_macro': recall(average='macro', **metric_params),
                f'{step_name}/recall_macro': recall(average='macro', **metric_params),
                
                f'{step_name}/loss': values['loss'],
                # f'{step_name}/lr': get_lr(optimizer),
                'epoch': epoch 
            }
        )
        
    return model


if __name__ == '__main__':
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = VGG(config=MODEL_CFG, num_classes=200).to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    run.define_metric("train/*", step_metric='epoch')
    run.define_metric("eval/*", step_metric="epoch")
    run.watch(model, log='all', log_freq=200)
    
    for epoch in range(1, EPOCHS):
        model.train()
        model = step(model, optimizer, criterion, train_loader, epoch)
        
        # torch.save(model.state_dict(), f'./models_state_dict/model_{epoch}.pt')
        # artifact.add_file(f'./models_state_dict/model_{epoch}.pt') 
        
        model.eval()
        _ = step(model, None, criterion, valid_loader, epoch, step_name='eval')
    
    
                    