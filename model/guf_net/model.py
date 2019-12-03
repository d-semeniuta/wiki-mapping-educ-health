"""
PyTorch convolution net for predicting health and education metrics using
Global Urban Footprint data

    author: Daniel Semeniuta (dsemeniu@cs.stanford.edu)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import GUFAfricaDataset
from .layers.guf_conv import GUFConv

class GUFNet(nn.Module):
    def __init__(self, task, params):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        super(GUFNet, self).__init__()
        if task not in ['imr', 'mated']:
            raise(ValueError('Incorrect Task'))
        self.task = task

        self.conv_net = GUFConv(256, params['conv_activation'])
        self.sigmoid_out = params['sigmoid_out']
        self.out_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv_net(x)

        if self.sigmoid_out and self.task == 'mated':
            return 3 * torch.sigmoid(self.out_layers(x))
        elif self.sigmoid_out:
            return torch.sigmoid(self.out_layers(x))
        else:
            return self.out_layers(x)

def main():
    gufdata = GUFAfricaDataset()
    params = {
        'lr': 8e-4,
        'batch_size': 16,
        'sigmoid_out': False,
        'conv_activation': 'relu',
        'num_epochs': 150
    }
    gufnet = GUFNet('imr', params)

    sample = gufdata[0]['image'].unsqueeze(0)
    print(sample, sample.shape)
    out = gufnet(sample)
    print(out, out.shape)

    bothnet = GUFNet('both', params)
    imr_out, mated_out = bothnet(sample)
    print(imr_out, mated_out)

if __name__ == '__main__':
    main()
