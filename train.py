"""
Train the model
"""

import argparse
import os
import json

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()
    param_loc = os.path.join(args.model_dir, 'params.json')
    params = json.load(param_loc)
    return args, params


def train(models, optimizers, loss_fns, dataloader, params):
    """Trains the model

    Parameters
    ----------
    models : dict(nn.Module)
        Dictionary containing IMR and MatEd models
    optimizers : dict(torch.optim)
        Description of parameter `optimizer`.
    loss_fns : dict(nn.modules.loss)
        Description of parameter `loss_fn`.
    dataloader : data.DataLoader

    Returns
    -------
    type
        Description of returned object.

    """
    model.train()
