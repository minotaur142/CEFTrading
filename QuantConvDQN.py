import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch 

class QuantConv(nn.Module):
    def __init__(self, shape, actions_n):
        super(QuantConv, self).__init__()
        self.actions_n = actions_n
        self.conv =nn.Sequential(
            nn.Conv1d(shape[0], 64, 4),
            nn.ReLU(),
            nn.Conv1d(64, 32, 2),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, actions_n*200)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(1)
        out = self.fc(out).view(-1, self.actions_n, 200)
        return out
