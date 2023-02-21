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

import cfg as CFG
from resnet import ResNet
from data.dataset import train_set, valid_set


os.environ["WANDB_SILENT"] = "true"


class Trainer(object):
    def __init__(self) -> None:        
        self.train_loader = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(valid_set, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4)
        
        self.model = ResNet(CFG.MODEL_CFG).to(CFG.DEVICE)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=CFG.LR)
        self.criterion = nn.CrossEntropyLoss()
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
        #                                                      max_lr=0.01, 
        #                                                      steps_per_epoch=len(self.train_loader) // BATCH_SIZE, 
        #                                                      epochs=EPOCHS)
        
        self.run = wandb.init(project="resnet_classifier")
        self.run.define_metric("train/*", step_metric='epoch')
        self.run.define_metric("eval/*", step_metric="epoch")
        self.run.watch(self.model, log='all', log_freq=50)
        self.artifact = wandb.Artifact('model', type='model')
        
    def fit(self):
        """
        Starts the training loop.
        """        
        
        for epoch in range(1, CFG.EPOCHS + 1):
            self.model.train()
            self.step(self.train_loader, epoch)
            
            if epoch == CFG.EPOCHS:
                model_path = os.path.join(os.path.dirname(__file__), f'./models_state_dict/model_{epoch}.pt')
                torch.save(self.model.state_dict(), model_path)
                self.artifact.add_file(model_path) 
            
            self.model.eval()
            self.step(self.valid_loader, epoch, step_name='eval')
        
        self.run.log_artifact(self.artifact)
    
    def get_lr(self) -> torch.FloatTensor:
        """
        Return current learning rate.

        Returns:
            torch.float: learning rate value
        """
        
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step(self, loader: DataLoader, epoch: int, step_name='train'):
        """
        Performs the training / evaluation step.

        Args:
            loader (DataLoader): train / eval loader
            epoch (int): epoch number
            step_name (str, optional): train - training step (grad enables), 
                                       eval - validation step. Defaults to 'train'.
        """     
           
        scaler = torch.cuda.amp.GradScaler(enabled=CFG.USE_AMP)
        
        values = {
            'loss': [],
            'predicts': [],
            'targets': []
        }
        
        with torch.inference_mode(mode=step_name == 'eval'):
            for batch in tqdm(loader, desc=step_name):
                images, targets = [item.to(CFG.DEVICE) for item in batch]
                
                with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                    output = self.model(images)
                    loss = self.criterion(output, targets)
    
                    values['loss'].append(loss.item())
                    values['predicts'].append(output.softmax(1))
                    values['targets'].append(targets)

                    if step_name == 'train':
                        self.optimizer.zero_grad()
                        
                        if CFG.USE_AMP:
                            scaler.scale(loss).backward()
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                            
                        # self.scheduler.step()
                
            values['predicts'] = torch.cat(values['predicts'], dim=0)
            values['targets'] = torch.cat(values['targets'], dim=0)
            
            metric_params = dict(
                preds=values['predicts'],
                target=values['targets'],
                task='multiclass', 
                num_classes=CFG.NUM_CLASSES
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
                    # f'{step_name}/lr': self.scheduler.get_last_lr(),
                    'epoch': epoch 
                }
            )

if __name__ == '__main__':
    Trainer().fit()
                    