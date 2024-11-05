import gc

import torch
import torch.nn.functional as F
import torchvision.models
from lightning import LightningModule
from torch import nn


class CNNLSTM(LightningModule):

    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.classifier = nn.Identity()

        self.lstm = nn.LSTM(input_size=1280, hidden_size=512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x_3d):
        batch_size, seq_len, c, h, w = x_3d.size()
        x_3d = x_3d.view(batch_size * seq_len, c, h, w)
        x = self.efficientnet(x_3d)
        x = x.view(batch_size, seq_len, -1)
        x = x.permute(1, 0, 2)

        out, (h_n, c_n) = self.lstm(x)
        x = self.fc1(torch.mean(out, dim=0))
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx) :
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('loss', loss, prog_bar=True)

        del x, y, y_hat
        gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log('val_loss', loss, prog_bar=True)

        del x, y
        gc.collect()

        return loss
