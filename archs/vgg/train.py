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


os.environ["WANDB_SILENT"] = "true"


class Trainer(object):
    def __init__(self) -> None:
        self.run = wandb.init(project="vgg_classifier")
        self.artifact = wandb.Artifact('model', type='model')
        
        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        self.valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        self.model = VGG(config=MODEL_CFG, num_classes=200).to(DEVICE)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()
        
        self.run.define_metric("train/*", step_metric='epoch')
        self.run.define_metric("eval/*", step_metric="epoch")
        self.run.watch(self.model, log='all', log_freq=50)
        
    def fit(self):
         for epoch in range(1, EPOCHS + 1):
            self.model.train()
            self.step(self.train_loader, epoch)
            
            # torch.save(self.model.state_dict(), f'./models_state_dict/model_{epoch}.pt')
            # self.artifact.add_file(f'./models_state_dict/model_{epoch}.pt') 
            
            self.model.eval()
            self.step(self.valid_loader, epoch, step_name='eval')
    
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step(self, loader, epoch, step_name='train') -> nn.Module:
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
                    output = self.model(images)
                    loss = self.criterion(output, targets)
    
                    values['loss'].append(loss.item())
                    values['predicts'].append(output.softmax(1))
                    values['targets'].append(targets)
                    
                    if step_name == 'train':
                        self.optimizer.zero_grad()
                        
                        if USE_AMP:
                            scaler.scale(loss).backward()
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                # break
            values['predicts'] = torch.cat(values['predicts'], dim=0)
            values['targets'] = torch.cat(values['targets'], dim=0)
            
            metric_params = dict(
                preds=values['predicts'],
                target=values['targets'],
                task='multiclass', 
                num_classes=200
            )

            self.run.log(
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
                    
                    f'{step_name}/loss': np.mean(values['loss']),
                    # f'{step_name}/lr': get_lr(optimizer),
                    'epoch': epoch 
                }
            )

if __name__ == '__main__':
    Trainer().fit()
                    