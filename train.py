"""
Train the model
"""

import argparse
import os
import json
import random

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

from scipy.stats import pearsonr, linregress

from tensorboardX import SummaryWriter

import numpy as np

import plotly.graph_objects as go

from tqdm import tqdm

from util.data import getDataLoaders
from model.combined.model import MultiModalNet
from model.guf_net.model import GUFNet
from model.doc_net.model import Doc2VecNet
from model.graph_net.model import Graph2VecNet


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'last'
    parser.add_argument('--use_graph', action='store_true', help='Use graph embeddings when training, default false')
    parser.add_argument('--overfit', action='store_true', help='Overfit to a training set')
    parser.add_argument('--vec_feature_path', default=None, help="File containing vec features")
    parser.add_argument('--guf_path', default=None, help="File containing guf images")

    args = parser.parse_args()
    param_loc = os.path.join(args.model_dir, 'params.json')
    with open(param_loc) as json_file:
        params = json.load(json_file)
    params['device'] =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['model_dir'] = args.model_dir
    return args, params

# def get_loss(out, task, labels, loss_fn):
#     imr, ed_score = labels
#     if task == 'imr':
#         loss = loss_fn(out, imr.unsqueeze(-1))
#     elif task == 'mated':
#         loss = loss_fn(out, ed_score.unsqueeze(-1))
#     else:
#         loss = loss_fn(out[0], imr.unsqueeze(-1)) + loss_fn(out[1], ed_score.unsqueeze(-1))
#     return loss

def model_forward(model, model_type, embs, imgs):
    if model_type == 'combined':
        return model.forward(embs, imgs)
    elif model_type == 'guf':
        return model.forward(imgs)
    else:
        return model.forward(embs)

def train_model(training_dict, train_loader, val_loader, writer, params):
    """Trains the model for a single epoch

    Parameters
    ----------
    models : dict(nn.Module)
        Dictionary containing IMR and MatEd models
    optimizers : dict(torch.optim)
        Dictionary containing IMR and MatEd optimizers
    loss_fns : dict(nn.modules.loss)
        Dictionary containing IMR and MatEd loss fns
    train_loader : data.DataLoader
    val_loader : data.DataLoader
    writer
        tensorboardX summary writer
    params : dict

    """
    # unwrap training dict
    num_epochs = params['num_epochs']
    models = training_dict['models']
    optimizers = training_dict['optims']
    loss_fns = training_dict['loss_fns']
    epoch = training_dict['epoch']  # in case loading from checkpoint

    for model in models.values():
        model.train()

    device = params['device']
    step = num_epochs * len(train_loader)
    total_batches = params['num_epochs'] * len(train_loader)
    best_corrs = {'imr': -1, 'mated': -1}

    with tqdm(total=total_batches) as progress_bar:
        while epoch < params['num_epochs']:
            epoch += 1 # one-indexed epochs
            progress_bar.set_postfix(epoch=epoch)
            for i, batch in enumerate(train_loader):
                step += 1
                imr, ed_score = batch['imr'].to(device), batch['ed_score'].to(device)
                embeddings, images = batch['emb'].to(device), batch['image'].to(device)
                labels = {'imr': imr, 'mated': ed_score}
                for task, model in models.items():
                    optimizer = optimizers[task]
                    # out = model.forward(embeddings, images)
                    out = model_forward(model, params['model_type'], embeddings, images)

                    loss_fn = loss_fns[task]
                    loss = loss_fn(out, labels[task])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('train/{}/loss'.format(task), loss.item(), step)
                progress_bar.update(1)
            # check evaluation step
            if epoch % params['eval_every'] == 0:
                (corrs, losses), _ = evaluate(models, val_loader, loss_fns, params)
                for task in corrs.keys():
                    writer.add_scalar('val/{}/r2'.format(task), corrs[task], epoch)
                    writer.add_scalar('val/{}/loss'.format(task), losses[task], epoch)

                for model in models.values():
                    model.train()

                progress_bar.set_postfix(
                    r2_imr=corrs['imr'],
                    r2_mated=corrs['mated'],
                    loss_imr=losses['imr'],
                    loss_mated=losses['mated'],
                )
                # save models
                for task, model in models.items():
                    out_dir = os.path.join(params['model_dir'], params['train_country'])
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)
                    dict_to_save = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizers[task].state_dict(),
                        'best_corr': best_corrs[task],
                        'task': task
                    }
                    if corrs[task] > best_corrs[task]:
                        # best by correlation
                        best_corrs[task] = corrs[task]
                        dict_to_save['best_corr'] = best_corrs[task]
                        best_out = os.path.join(out_dir, '{}.best.pth'.format(task))
                        torch.save(dict_to_save, best_out)
                    last_out = os.path.join(out_dir, '{}.last.pth'.format(task))
                    torch.save(dict_to_save, last_out)
    return models

def evaluate(models, val_loader, loss_fns, params):
    for model in models.values():
        model.eval()
    device = params['device']
    with torch.no_grad():
        ins, outs = {'imr': [], 'mated': []}, {'imr': [], 'mated': []}
        for batch in val_loader:
            imr, ed_score = batch['imr'].to(device), batch['ed_score'].to(device)
            embeddings, images = batch['emb'].to(device), batch['image'].to(device)
            for task, model in models.items():
                # out = model.forward(embeddings, images)
                out = model_forward(model, params['model_type'], embeddings, images)
                outs[task].append(out.detach().squeeze(-1))
            ins['imr'].append(imr)
            ins['mated'].append(ed_score)

        for task in outs.keys():
            # cat into single array
            ins[task] = torch.cat(ins[task])
            outs[task] = torch.cat(outs[task])
        corrs = {}
        losses = {}
        for task in outs.keys():
            corrs[task] = pearsonr(ins[task].numpy(), outs[task].numpy())[0]
            loss_fn = loss_fns[task]
            losses[task] = loss_fn(ins[task], outs[task]).item()

    return (corrs, losses), (ins, outs)

def chooseModel(task, args, params):
    models = ['combined', 'guf', 'graph', 'doc']
    model = params['model_type']

    if model == 'combined':
        return MultiModalNet(params, args.use_graph)
    elif model == 'guf':
        return GUFNet(task, params)
    elif model == 'graph':
        return Graph2VecNet(task, params)
    elif model == 'doc':
        return Doc2VecNet(task, params)
    else:
        raise ValueError('Incorrect Model type')

def loadModels(args, params):
    models = {}
    optims = {}
    best_corrs = {}
    loss_fns = {}
    epoch = 0
    for task in ['imr', 'mated']:
        # model = MultiModalNet(params, args.use_graph)
        model = chooseModel(task, args, params).to(params['device'])
        curr_optim = optim.Adam(
            model.parameters(), lr=params['lr'], betas=(params['b1'], params['b2'])
        )
        best_corr = -1
        if args.restore_file is not None:
            if args.restore_file is not 'last':
                raise(ValueError("Can't load from best with current training setup"))
            cp_loc = os.path.join(args.model_dir, '{}.last.pth'.format(task))
            cp = torch.load(cp_loc)
            model.load_state_dict(cp['state_dict'])
            optim.load_state_dict(cp['optim_dict'])
            best_corr = cp['best_corr']
            epoch = cp['epoch']
        models[task] = model
        optims[task] = curr_optim
        best_corrs[task] = best_corr
        loss_fns[task] = nn.MSELoss()

    training_dict = {
        'models': models,
        'optims': optims,
        'best_corrs': best_corrs,
        'loss_fns': loss_fns,
        'epoch': epoch
    }
    return training_dict

def train_loop(args, params):
    countries = params['countries']
    country_opts = countries + ['all']
    print('Generating data loaders...')
    data_loaders = getDataLoaders(countries, args.guf_path, args.vec_feature_path,
                                    params['batch_size'], use_graph=args.use_graph,
                                    overfit=args.overfit)
    if args.overfit:
        print('Overfitting...')
        country_opts = countries[0]
    for train in country_opts:
        print('\nTraining on {}...'.format(train))
        train_loader = data_loaders[train]['train']
        val_loader = data_loaders[train]['val']
        writer_dir = os.path.join(args.model_dir, 'tb', train)
        writer = SummaryWriter(writer_dir)
        training_dict = loadModels(args, params)
        params['train_country'] = train
        # loss_fns = training_dict['loss_fns']
        models = train_model(training_dict, train_loader, val_loader, writer, params)

        print('Model trained in {} results:'.format(train))
        log_file_loc = os.path.join(args.model_dir, '{}_train.txt'.format(train))
        with open(log_file_loc, 'w') as log_file:
            for val in country_opts:
                if val == 'all' and train != 'all':
                    val_loader = data_loaders[train]['others']['val']
                else:
                    val_loader = data_loaders[val]['val']
                plot_info = {
                    'save_dir' : os.path.join(args.model_dir, 'plots'),
                    'title' : 'Train in {}, Val in {}'.format(train, val)
                }
                if not os.path.exists(plot_info['save_dir']):
                    os.makedirs(plot_info['save_dir'])
                (corrs, losses), (ins, outs) = evaluate_model(models, val_loader, training_dict['loss_fn'])
                plotPreds(ins, outs, corrs, plot_info)
                log_file.write('Validated in {}\n'.format(val))
                log_file.write('Separate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}\n'.format(corrs['imr'], corrs['mated']))
                print('\tValidated in {}'.format(val))
                print('\tSeparate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['imr'], corrs['mated']))

def main():
    args, params = parseArgs()
    # countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    train_loop(args, params)


if __name__ == '__main__':
    main()
