import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Graph2VecAfricaDataset

class Graph2VecNet(nn.Module):
    def __init__(self, task, params):
        super(Graph2VecNet, self).__init__()

        if task not in ['imr', 'mated', 'both']:
            raise(ValueError('Incorrect Task'))
        self.task = task

        self.sigmoid_out = params['sigmoid_out']

        out_layers = [
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ]
        if self.task == 'both':
            self.imr_out = nn.Sequential(
                nn.Linear(300, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.mated_out = nn.Sequential(
                nn.Linear(300, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            self.out_layers = nn.Sequential(*out_layers)

    def forward(self, x):
        if self.sigmoid_out and self.task == 'mated':
            return 3 * F.sigmoid(self.out_layers(x))
        elif self.sigmoid_out and self.task == 'both':
            return (
                F.sigmoid(self.imr_out(x)),
                3 * F.sigmoid(self.mated_out(x))
            )
        elif self.task == 'both':
            return (
                self.imr_out(x),
                self.mated_out(x)
            )
        else:
            return F.sigmoid(self.out_layers(x)) if self.sigmoid_out else self.out_layers(x)
