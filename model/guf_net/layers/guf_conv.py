import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GUFConv(nn.Module):
    def __init__(self, outsize, conv_activation):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        super(GUFConv, self).__init__()
        if conv_activation == 'relu':
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
        elif conv_activation == 'sigmoid':
            self.conv_net = nn.Sequential(
                nn.Conv2d(1, 3, 5),
                nn.Sigmoid(),
                nn.Conv2d(3, 5, 5),
                nn.Sigmoid(),
                nn.Flatten(),
                nn.Linear(15680, 512),
                nn.ReLU(),
                nn.Linear(512, outsize),
                nn.ReLU()
            )
        else:
            self.conv_net = nn.Sequential(
                nn.Conv2d(1, 3, 5),
                nn.Conv2d(3, 5, 5),
                nn.Flatten(),
                nn.Linear(15680, 512),
                nn.ReLU(),
                nn.Linear(512, outsize),
                nn.ReLU()
            )



    def forward(self, x):
        return self.conv_net(x)
