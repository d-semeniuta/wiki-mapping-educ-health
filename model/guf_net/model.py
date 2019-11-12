"""
PyTorch convolution net for predicting health and education metrics using
Global Urban Footprint data

    author: Daniel Semeniuta (dsemeniu@cs.stanford.edu)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import GUFAfricaDataset
from layers.guf_conv import GUFConv

class GUFNet(nn.Module):
    def __init__(self, task, params):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        super(GUFNet, self).__init__()
        if task not in ['imr', 'mated', 'both']:
            raise(ValueError('Incorrect Task'))
        self.task = task


        self.conv_net = GUFConv(256, params['conv_activation'])
        self.sigmoid_out = params['sigmoid_out']
        out_layers = [
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ]
        if self.sigmoid_out:
            out_layers.append(nn.Sigmoid())
        self.out_layers = nn.Sequential(*out_layers)

    def forward(self, x):
        x = self.conv_net(x)
        if self.sigmoid_out and self.task == 'mated':
            return 4 * self.out_layers(x)
        else:
            return self.out_layers(x)

def main():
    gufdata = GUFAfricaDataset()
    gufnet = GUFNet('imr')

    sample = gufdata[0]['image'].unsqueeze(0)
    print(sample, sample.shape)
    out = gufnet(sample)
    print(out, out.shape)

if __name__ == '__main__':
    main()
