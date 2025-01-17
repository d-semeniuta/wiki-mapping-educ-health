"""
Train the model
"""

import argparse
import os, pdb
import json
import random

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

from scipy.stats import pearsonr, linregress
from sklearn.metrics import r2_score

from tensorboardX import SummaryWriter

import numpy as np

import plotly.graph_objects as go

from tqdm import tqdm

from util.data import getDataLoaders
from util.utils import plotPreds
from model.combined.model import MultiModalNet
from model.graph_doc_guf.model import AllThreeNet
from model.guf_net.model import GUFNet
from model.doc_net.model import Doc2VecNet
from model.graph_net.model import Graph2VecNet


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'last'
    # parser.add_argument('--use_graph', action='store_true', help='Use graph embeddings when training, default false')
    parser.add_argument('--overfit', action='store_true', help='Overfit to a training set')
    parser.add_argument('--vec_feature_path', default=None, help="File containing vec features")
    parser.add_argument('--guf_path', default=None, help="File containing guf images")
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')

    args = parser.parse_args()
    param_loc = os.path.join(args.model_dir, 'params.json')
    with open(param_loc) as json_file:
        params = json.load(json_file)
    params['device'] =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['model_dir'] = args.model_dir
    return args, params

def model_forward(model, model_type, *, doc_embs, graph_embs, imgs, params):
    if model_type == 'all_three':
        return model.forward(graph_embs=graph_embs, doc_embs=doc_embs, imgs=imgs)
    if model_type == 'combined' and params['use_graph']:
        return model.forward(graph_embs, imgs)
    elif model_type == 'combined':
        return model.forward(doc_embs, imgs)
    elif model_type == 'guf':
        return model.forward(imgs)
    elif params['use_graph']:
        return model.forward(graph_embs)
    else:
        return model.forward(doc_embs)

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
    best_corrs = training_dict['best_corrs']
    best_r2s = training_dict['best_r2s']

    for model in models.values():
        model.train()

    device = params['device']
    step = epoch * len(train_loader)
    total_batches = params['num_epochs'] * len(train_loader)

    with tqdm(total=total_batches) as progress_bar:
        while epoch < params['num_epochs']:
            epoch += 1 # one-indexed epochs
            for i, batch in enumerate(train_loader):
                step += 1
                imr, ed_score = batch['imr'].to(device), batch['ed_score'].to(device)
                doc_embs, graph_embs, images = batch['doc_emb'].to(device), batch['graph_emb'].to(device), batch['image'].to(device)
                labels = {'imr': imr, 'mated': ed_score}
                for task, model in models.items():
                    optimizer = optimizers[task]
                    out = model_forward(model, params['model_type'], doc_embs=doc_embs,
                                            graph_embs=graph_embs, imgs=images, params=params)

                    loss_fn = loss_fns[task]
                    loss = loss_fn(out, labels[task].unsqueeze(-1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('train/{}/loss'.format(task), loss.item(), step)
                progress_bar.update(1)
            # check evaluation step
            if epoch % params['eval_every'] == 0:
                (corrs, r2s, losses), _ = evaluate(models, val_loader, loss_fns, params)
                for task in corrs.keys():
                    writer.add_scalar('val/{}/corr'.format(task), corrs[task], epoch)
                    writer.add_scalar('val/{}/r2'.format(task), r2s[task], epoch)
                    writer.add_scalar('val/{}/loss'.format(task), losses[task], epoch)

                for model in models.values():
                    model.train()

                progress_bar.set_postfix(
                    pearsonr_imr=corrs['imr'],
                    pearsonr_mated=corrs['mated'],
                    r2_imr=r2s['imr'],
                    r2_mated=r2s['mated'],
                    loss_imr=losses['imr'],
                    loss_mated=losses['mated'],
                    epoch=epoch
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
                        'best_r2': best_r2s[task],
                        'task': task
                    }
                    if corrs[task] > best_corrs[task]:
                        # best by correlation
                        best_corrs[task] = corrs[task]
                        dict_to_save['best_corr'] = best_corrs[task]
                        best_out = os.path.join(out_dir, '{}.bestcorr.pth'.format(task))
                        torch.save(dict_to_save, best_out)
                    if r2s[task] > best_r2s[task]:
                        # best by r2
                        best_r2s[task] = r2s[task]
                        dict_to_save['best_r2'] = best_r2s[task]
                        best_out = os.path.join(out_dir, '{}.bestr2.pth'.format(task))
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
            doc_embs, graph_embs = batch['doc_emb'].to(device), batch['graph_emb'].to(device)
            images = batch['image'].to(device)
            for task, model in models.items():
                out = model_forward(model, params['model_type'], doc_embs=doc_embs,
                                        graph_embs=graph_embs, imgs=images, params=params)
                outs[task].append(out.detach().squeeze(-1))
            ins['imr'].append(imr)
            ins['mated'].append(ed_score)

        for task in outs.keys():
            # cat into single array
            ins[task] = torch.cat(ins[task])
            outs[task] = torch.cat(outs[task])
        corrs = {}
        losses = {}
        r2s = {}
        for task in outs.keys():
            corrs[task] = pearsonr(ins[task].numpy(), outs[task].numpy())[0]
            r2s[task] = r2_score(ins[task].numpy(), outs[task].numpy())
            loss_fn = loss_fns[task]
            losses[task] = loss_fn(ins[task], outs[task]).item()

    return (corrs, r2s, losses), (ins, outs)

def chooseModel(task, args, params):
    models = ['combined', 'guf', 'graph', 'doc', 'all_three']
    model = params['model_type']

    if model == 'combined':
        return MultiModalNet(params, params['use_graph'])
    elif model == 'guf':
        return GUFNet(task, params)
    elif model == 'graph':
        return Graph2VecNet(task, params)
    elif model == 'doc':
        return Doc2VecNet(task, params)
    elif model == 'all_three':
        return AllThreeNet(task, params)
    else:
        raise ValueError('Incorrect Model type')

def loadModels(train_country, args, params):
    models = {}
    optims = {}
    best_corrs = {}
    best_r2s = {}
    loss_fns = {}
    epoch = 0
    for task in ['imr', 'mated']:
        # model = MultiModalNet(params, args.use_graph)
        model = chooseModel(task, args, params).to(params['device'])
        curr_optim = optim.Adam(
            model.parameters(), lr=params['lr'],
            betas=(params['b1'], params['b2']),
            weight_decay=params['weight_decay']
        )
        best_corr = -1
        best_r2 = float('-inf')
        if args.restore_file is not None:
            if args.restore_file != 'last':
                raise(ValueError("Can't load from best with current training setup"))
            cp_loc = os.path.join(args.model_dir, train_country, '{}.last.pth'.format(task))
            cp = torch.load(cp_loc)
            model.load_state_dict(cp['state_dict'])
            curr_optim.load_state_dict(cp['optim_dict'])
            best_corr = cp['best_corr']
            epoch = cp['epoch']
        models[task] = model
        optims[task] = curr_optim
        best_corrs[task] = best_corr
        best_r2s[task] = best_r2
        loss_fns[task] = nn.MSELoss()

    training_dict = {
        'models': models,
        'optims': optims,
        'best_corrs': best_corrs,
        'best_r2s': best_r2s,
        'loss_fns': loss_fns,
        'epoch': epoch
    }
    return training_dict

def train_loop(args, params):
    countries = params['countries']
    country_opts = countries + ['all']
    print('Generating data loaders...')
    data_loaders = getDataLoaders(countries, args.guf_path, args.vec_feature_path,
                                    params['batch_size'], args.model_dir,
                                    use_graph=params['use_graph'], overfit=args.overfit)
    if args.overfit:
        print('Overfitting...')
        country_opts = [countries[0]]
    for train in country_opts:
        print('\nTraining on {}...'.format(train))
        train_loader = data_loaders[train]['train']
        val_loader = data_loaders[train]['val']
        writer_dir = os.path.join(args.model_dir, 'tb', train)
        writer = SummaryWriter(writer_dir)
        training_dict = loadModels(train, args, params)
        params['train_country'] = train
        models = train_model(training_dict, train_loader, val_loader, writer, params)

        print('Model trained in {} results:'.format(train))
        log_file_loc = os.path.join(args.model_dir, '{}_train.log'.format(train))
        with open(log_file_loc, 'w') as log_file:
            for val in country_opts:
                if val == 'all' and train != 'all':
                    val_loader = data_loaders[train]['others']['val']
                else:
                    val_loader = data_loaders[val]['val']
                plot_info = {
                    'save_dir' : os.path.join(args.model_dir, train, 'plots', val),
                    'title' : 'Train in {}, Val in {}'.format(train, val)
                }
                if not os.path.exists(plot_info['save_dir']):
                    os.makedirs(plot_info['save_dir'])
                (corrs, r2s, losses), (ins, outs) = evaluate(models, val_loader, training_dict['loss_fns'], params)
                plotPreds(ins, outs, r2s, plot_info)
                log_file.write('Validated in {}\n'.format(val))
                log_file.write('Separate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}\n'.format(corrs['imr'], corrs['mated']))
                log_file.write('\tIMR r2: {:.3f}\t\tMatEd r2: {:.3f}'.format(corrs['imr']**2, corrs['mated']**2))
                log_file.write('\tIMR R2: {:.3f}\t\tMatEd R2: {:.3f}\n'.format(r2s['imr'], r2s['mated']))
                print('\tValidated in {}'.format(val))
                print('\tSeparate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['imr'], corrs['mated']))
                print('\t\tIMR r2: {:.3f}\t\tMatEd r2: {:.3f}'.format(corrs['imr']**2, corrs['mated']**2))
                print('\t\t IMR R2: {:.3f}\t\tMatEd R2: {:.3f}\n'.format(r2s['imr'], r2s['mated']))

def evaluate_loop(args, params):
    if args.restore_file is None:
        raise ValueError('Need to load model to evaluate')
    countries = params['countries']
    country_opts = countries + ['all']
    print('Generating data loaders...')
    data_loaders = getDataLoaders(countries, args.guf_path, args.vec_feature_path,
                                    params['batch_size'], args.model_dir,
                                    use_graph=params['use_graph'], overfit=args.overfit)
    for train in country_opts:
        print('\nTrained on {}...'.format(train))
        val_loader = data_loaders[train]['val']
        training_dict = loadModels(train, args, params)
        models = training_dict['models']
        params['train_country'] = train
        print('Model trained in {} results:'.format(train))
        log_file_loc = os.path.join(args.model_dir, '{}_eval.txt'.format(train))
        with open(log_file_loc, 'w') as log_file:
            for val in country_opts:
                if val == 'all' and train != 'all':
                    val_loader = data_loaders[train]['others']['val']
                else:
                    val_loader = data_loaders[val]['val']
                plot_info = {
                    'save_dir' : os.path.join(args.model_dir, train, 'plots', val),
                    'title' : 'Train in {}, Val in {}'.format(train, val)
                }
                if not os.path.exists(plot_info['save_dir']):
                    os.makedirs(plot_info['save_dir'])
                (corrs, r2s, losses), (ins, outs) = evaluate(models, val_loader, training_dict['loss_fns'], params)
                plotPreds(ins, outs, r2s, plot_info)
                log_file.write('Validated in {}\n'.format(val))
                log_file.write('Separate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}\n'.format(corrs['imr'], corrs['mated']))
                log_file.write('\tIMR r2: {:.3f}\t\tMatEd r2: {:.3f}\n'.format(corrs['imr']**2, corrs['mated']**2))
                log_file.write('\tIMR R2: {:.3f}\t\tMatEd R2: {:.3f}\n'.format(r2s['imr'], r2s['mated']))
                print('\tValidated in {}'.format(val))
                print('\tSeparate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['imr'], corrs['mated']))
                print('\t\tIMR r2: {:.3f}\t\tMatEd r2: {:.3f}\n'.format(corrs['imr']**2, corrs['mated']**2))
                print('\t\t IMR R2: {:.3f}\t\tMatEd R2: {:.3f}\n'.format(r2s['imr'], r2s['mated']))



def main():
    args, params = parseArgs()
    if args.eval:
        evaluate_loop(args, params)
    else:
        train_loop(args, params)


if __name__ == '__main__':
    main()
