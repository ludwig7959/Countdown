import torch.nn.functional as F
import torchvision.models
from lightning import LightningModule
from torch import nn

from minGRU import minGRU


class CNNLSTM(LightningModule):

    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
        self.efficientnet.fc = nn.Sequential(nn.Linear(self.efficientnet.fc.in_features, 512))
        self.lstm = minGRU(512, 1.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            x = self.efficientnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x, hidden, return_next_prev_hidden=True)
        x = self.fc1(out[:, -1:, :])
        x = F.relu(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx) :
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
