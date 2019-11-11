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
    def __init__(self, task):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        super(GUFNet, self).__init__()
        if task not in ['imr', 'mated', 'both']:
            Raise(ValueError('Incorrect Task'))


        self.conv_net = GUFConv(256)
        self.out_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_net(x)
        # return self.out_layers(x)
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
