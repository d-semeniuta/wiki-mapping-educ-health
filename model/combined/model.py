import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../../')
from model.guf_net.layers.guf_conv import GUFConv

class MultiModalNet(nn.Module):
    def __init__(self, params, use_graph=False):
        super(MultiModalNet, self).__init__()

        img_emb_size = 256
        self.guf_net = GUFConv(img_emb_size, params['conv_activation'])

        # magic numbers, can we do better?
        wiki_emb_size_in = 300 if use_graph else 3010
        wiki_emb_size_out = 256
        self.wiki_net = nn.Sequential(
            nn.Linear(wiki_emb_size_in, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, wiki_emb_size_out), nn.ReLU()
        )

        self.out_layers = nn.Sequential(
            nn.Linear(img_emb_size+wiki_emb_size_out, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, embs, images):
        img_emb = self.guf_net(images) # (batch_size, img_emb_size)
        wiki_emb = self.wiki_net(embs) # (batch_size, wiki_emb_size)
        out = torch.cat([img_emb, wiki_emb], 1)
        out = self.out_layers(out)
        return out
