from typing import Union, List
import pickle
import os
import timm.scheduler
import torch
import pandas as pd
from tqdm import tqdm,trange
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import numpy as np
from backbones import iresnet
SCORE_RANGE = 45
from timm.optim import Lookahead

class VideoFeatureExtractor(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.model = iresnet.iresnet50(pretrained=False)
        if args.pretrain=='webface':
            self.model.load_state_dict(torch.load('/usr/local/wywconda/webface_r50.pth'))


        # 创建额外的全连接层
        # print(self.model)
    def forward(self, x):
        x = self.model(x)
        return x


class VideoRegressionModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.feature_extractor = VideoFeatureExtractor(args)
        self.mse = nn.MSELoss()
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()
        self.learning_rate = args.learning_rate

        self.validation_step_outputs1 = []
        self.validation_step_outputs2 = []
        self.validation_step_outputs3 = []
        self.loss_step=[]
        self.features = []
        self.additional_layers = nn.Sequential(
            nn.Linear(self.feature_extractor.model.fc.out_features, 512),  # 第一个全连接层，512个神经元
            nn.LeakyReLU(),  # ReLU激活函数
            nn.Linear(512, 128),  # 第二个全连接层，128个神经元
            nn.LeakyReLU(),  # ReLU激活函数
            nn.Dropout(args.dropout_rate),
            nn.Linear(128, 1),  # 第二个全连接层，128个神经元
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        x=self.additional_layers(feature)
        return x,feature

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        y_hat,_ = self(x)

        loss = F.mse_loss(y_hat, y.reshape((-1, 1)))
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.lookahead.zero_grad()
        self.lookahead.step()
        return loss


    def validation_step(self, batch, batch_idx):
        self._test_or_validation_step(batch,'val')

    def test_step(self, batch, batch_idx):
        self._test_or_validation_step(batch,'test')

    def _test_or_validation_step(self, batch,stage):
        x, y, z = batch
        y_hat,feature = self(x)
        loss = F.mse_loss(y_hat, y.reshape((-1, 1)))
        self.log('{}_loss'.format(stage), loss, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=y.shape[0])
        self.validation_step_outputs1.append(y_hat)
        self.validation_step_outputs2.append(y)
        self.validation_step_outputs3.append(z)
        if stage == 'test':
            self.features.append(feature)
        self.loss_step.append(loss.item())

    def _calculate_metrics(self, stage):
        name = self.validation_step_outputs3
        names = []
        for i in range(len(name)):
            names =names + [name.split('\\')[-2] for name in list(name[i])]

        y_hat = torch.cat(self.validation_step_outputs1, dim=0)[:, 0]
        y = torch.cat(self.validation_step_outputs2, dim=0)

        name1 = np.array(names)

        meanpre = []
        meanlable = []
        for name in list(set(names)):
            indexs = np.where(name1 == name)
            if y_hat[indexs].shape[0]>5:
                mid=torch.tensor(sorted(y_hat[indexs])[int(y_hat[indexs].shape[0]*self.args.remove_rate):-int(y_hat[indexs].shape[0]*self.args.remove_rate)])
                meanpre.append(mid[~torch.isnan(mid)].mean().item())
            else:
                meanpre.append(torch.tensor(y_hat[indexs]).mean().item())
            meanlable.append(y[indexs].mean().item())

        mae = self.mean_absolute_error(torch.tensor(meanpre), torch.tensor(meanlable))
        rmse = torch.sqrt(self.mean_squared_error(torch.tensor(meanpre), torch.tensor(meanlable)))
        # if stage=='test':
        # pd.DataFrame([meanpre, meanlable]).T.to_csv(
        #     os.path.join(self.logger.log_dir, 'pre.csv'), index=False)
        self.log('{}_mae_epoch'.format(stage), mae.item(), on_epoch=True, prog_bar=True, sync_dist=True,logger=True)
        self.log('{}_rmse_epoch'.format(stage), rmse, on_epoch=True, prog_bar=True, sync_dist=True,logger=True)
        self.log('{}_loss_epoch'.format(stage), np.array(self.loss_step).mean(), on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        self.validation_step_outputs1.clear()
        self.validation_step_outputs2.clear()
        self.validation_step_outputs3.clear()
        self.loss_step.clear()
        self.features.clear()
    def on_test_epoch_end(self) -> None:
        self._calculate_metrics(stage='test')
    def on_validation_epoch_end(self) -> None:
        self._calculate_metrics(stage='val')


    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                               min_lr=1e-7)
        self.lookahead = Lookahead(optimizer, k=5, alpha=0.5)

        # optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate,weight_decay=self.args.weight_decay)
        # optimizer = Lookahead(optimizer, k=5,alpha=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs,eta_min=1e-7)
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
