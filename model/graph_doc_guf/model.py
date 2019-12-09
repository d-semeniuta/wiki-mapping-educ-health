import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../../')
from model.guf_net.layers.guf_conv import GUFConv

class AllThreeNet(nn.Module):
    def __init__(self, params):
        super(MultiModalNet, self).__init__()

        img_emb_size = 128
        self.guf_net = GUFConv(img_emb_size, params['conv_activation'])

        # magic numbers, can we do better?
        doc_emb_in = 3010
        doc_emb_out = 128
        graph_emb_in = 300
        graph_emb_out = 128

        def wiki_emb_block(emb_in, emb_out):
            return [
                nn.Linear(emb_in, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, emb_out), nn.ReLU()
            ]

        self.doc_net = nn.Sequential(*wiki_emb_block(doc_emb_in, doc_emb_out))
        self.graph_net = nn.Sequential(*wiki_emb_block(graph_emb_in, graph_emb_out))
        self.out_layers = nn.Sequential(
            nn.Linear(img_emb_size+wiki_emb_size_out*2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, *, graph_embs, doc_embs, images):
        img_emb = F.relu(self.guf_net(images)) # (batch_size, img_emb_size)
        doc_emb = self.doc_net(doc_embs) # (batch_size, wiki_emb_size)
        graph_emb = self.graph_net(graph_embs) # (batch_size, wiki_emb_size)
        out = torch.cat([img_emb, doc_emb, graph_emb], 1)
        out = self.out_layers(out)
        return out
