import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
import mlflow
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import DataLoader
from mlflow import log_metric, log_param, log_artifacts
from torchmetrics import AUROC, Accuracy, Recall, Precision, F1Score

from vgg import VGG
from data.dataset import train_set, valid_set
from cfg import DEVICE, BATCH_SIZE, USE_AMP, EPOCHS, LR, MODEL_CFG


class VGGpl(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        self.model = VGG(**kwargs)
        
        common_params = dict(
            task='multiclass', 
            num_classes=200
        )
        
        self.metric = {
            'val_acc': Accuracy(**common_params),
            'val_auroc_macro': AUROC(average='macro', **common_params),
            'val_f1_micro': F1Score(average='micro', **common_params),
            'val_f1_macro': F1Score(average='macro', **common_params),

            'train_acc': Accuracy(**common_params),
            'train_auroc_macro': AUROC(average='macro', **common_params),
            'train_f1_micro': F1Score(average='micro', **common_params),
            'train_f1_macro': F1Score(average='macro', **common_params),
        }
        
        for k, v in self.metric.items():
            self.metric[k] = self.metric[k].to('cuda')
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        output = self.model(images)
        self.predicts = output.softmax(1).argmax(1)
        loss = nn.functional.cross_entropy(output, targets)
        
        self.metric['train_acc'].update(self.predicts, targets)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer
    
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        
        output = self.model(images)
        self.predicts = output.softmax(1).argmax(1)
        loss = nn.functional.cross_entropy(output, targets)
        

model = VGGpl({'config': MODEL_CFG, 'num_classes': 200})

trainer = pl.Trainer(accelerator='gpu',
                     check_val_every_n_epoch=1,
                     limit_train_batches=3,
                     limit_val_batches=3,
                     log_every_n_steps=1,
                     max_epochs=2,
                     profiler='simple'
                     )
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

# def step(model, optimizer, criterion, loader, step_name='train') -> nn.Module:
#     scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
#     values = {
#         'loss': [],
#         'predicts': [],
#         'targets': []
#     }

#     with torch.inference_mode(mode=step_name == 'eval'):
#         for batch in tqdm(loader, desc=step_name):
#             images, targets = [item.to(DEVICE) for item in batch]

#             with torch.cuda.amp.autocast(enabled=USE_AMP):
#                 output = model(images)
#                 predicts = output.softmax(1).argmax(1)
#                 loss = criterion(output, targets)
   
#                 values['loss'].append(loss.item())
#                 values['predicts'].append(predicts)
#                 values['targets'].append(targets)
                
#                 if step_name == 'train':
#                     optimizer.zero_grad()
                    
#                     if USE_AMP:
#                         scaler.scale(loss).backward()
#                         scaler.step(optimizer)
#                         scaler.update()
#                     else:
#                         loss.backward()
#                         optimizer.step()
#             break
#         values['predicts'] = torch.cat(values['predicts'], dim=0)
#         values['targets'] = torch.cat(values['targets'], dim=0)
        
#         metrics = {
#             f'{step_name} acc': float(accuracy(values['predicts'], values['targets'], task='multiclass', num_classes=200)),
#             f'{step_name} f1_micro': float(f1_score(values['predicts'], values['targets'], task='multiclass', average='micro', num_classes=200)),
#         }
        
#         model = None
        
#     return metrics


# if __name__ == '__main__':
#     mlflow.set_experiment('vgg_train')
    
#     train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
#     valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
#     model = VGG(config=MODEL_CFG, num_classes=200).to(DEVICE)
#     optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
#     criterion = nn.CrossEntropyLoss()
    
#     mlflow.start_run()
        
#     for epoch in range(EPOCHS):
#         model.train()
#         metrics = step(model, optimizer, criterion, train_loader)
#         mlflow.log_metrics(metrics, step=epoch)
#         model.eval()
#         metrics = step(model, None, criterion, valid_loader, step_name='eval')
    
    # mlflow.end_run()
                    