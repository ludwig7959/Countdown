import gc

import torch
import torch.nn.functional as F
import torchvision.models
from lightning import LightningModule
from torch import nn

from minGRU import minGRU


class CNNLSTM(LightningModule):

    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.classifier = nn.Sequential(nn.Linear(1280, 512))
        self.lstm = minGRU(512, 1.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_3d):
        # x_3d: (batch_size, seq_len, 3, 224, 224)
        features = []
        for t in range(x_3d.size(1)):
            x = self.efficientnet(x_3d[:, t, :, :, :])  # (batch_size, 512)
            features.append(x)

        features = torch.stack(features, dim=0)  # (seq_len, batch_size, 512)
        out = self.lstm(features)  # LSTM에 시퀀스 전체를 입력
        x = self.fc1(out[-1])  # 마지막 타임스텝의 출력 사용
        x = F.relu(x)
        x = self.fc2(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx) :
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('loss', loss, prog_bar=True)

        del x, y, y_hat
        gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('val_loss', loss, prog_bar=True)

        del x, y
        gc.collect()

        return loss
