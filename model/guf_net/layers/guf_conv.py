import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GUFConv(nn.Module):
    def __init__(self, outsize):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        super(GUFConv, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 3, 5),
            nn.ReLU(),
            nn.Conv2d(3, 5, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(15680, 512),
            nn.ReLU(),
            nn.Linear(512, outsize),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_net(x)
