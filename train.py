"""
Train the model
"""

import argparse
import os
import json

import torch

from scipy.stats import pearsonr

from util.data import getDataLoaders
from model.combined.model import MultiModalNet

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'last'
    parser.add_argument('--use_graph', action='store_true', help='Use graph embeddings when training, default false')
    parser.add_argument('--vec_feature_path', default=None, help="File containing vec features")

    args = parser.parse_args()
    param_loc = os.path.join(args.model_dir, 'params.json')
    params = json.load(param_loc)
    params['device'] =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['model_dir'] = args.model_dir
    return args, params


def train(training_dict, loss_fns, train_loader, val_loader, writer, params):
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
    for model in models.values():
        model.train()
    device = params['device']
    step = epoch * len(train_loader)
    total_batches = params['num_epochs'] * len(train_loader)
    best_corrs = {'imr': -1, 'mated': -1}
    with tqdm(total=total_batches) as progress_bar:
        epoch += 1
        while epoch != params['num_epochs']:
            for i, batch in enumerate(train_loader):
                step += 1
                images, imr, ed_score = batch['image'].to(device), batch['imr'].to(device), batch['ed_score'].to(device)
                labels = {'imr': imr, 'mated': ed_score}
                for task, model in models.items():
                    optimizer = optimizers[task]
                    out = model.forward(images)
                    loss = loss_fn(out, labels[task])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar('train/{}/loss'.format(task), loss.item(), step)
                progress_bar.update(1)
        # check evaluation step
        if epoch % params['eval_every'] == 0:
            (corrs, losses), _ = evaluate(models, val_loader, loss_fns, params, writer=writer)
            for task in in corrs.keys():
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
                out_dir = os.path.join(params['model_dir'], params['country'])
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
            images, imr, ed_score = batch['image'].to(device), batch['imr'].to(device), batch['ed_score'].to(device)
            for task, model in models.items():
                out = model.forward(images)
                outs[task].append(out.detach.squeeze(-1))
            ins['imr'].append(imr)
            ins['mated'].append(ed_score)

        for task in outs.keys():
            # cat into single array
            ins[task] = torch.cat(ins[task]).numpy()
            outs[task] = torch.cat(outs[task]).numpy()
        corrs = {}
        losses = {}
        for task in outs.keys():
            corrs[task] = pearsonr(ins[task], outs[task])[0]
            losses[task] = get_loss[task](torch.tensor(ins[task].to(device)),
                torch.tensor(outs[task].to(device))).item()

    return (corrs, losses), (ins, outs)

def loadModels(args, params):
    models = {}
    optims = {}
    best_corrs = {}
    loss_fns = {}
    epoch = 0
    for task in ['imr', 'mated']:
        model = MultiModalNet(params, args.use_graph)
        optim = optim.Adam(
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
        optims[task] = optim
        best_corrs[task] = best_corr
        loss_fn[task] = nn.MSELoss()

    training_dict = {
        'models': models,
        'optims': optims,
        'best_corrs': best_corrs,
        'loss_fns': loss_fns,
        'epoch': epoch
    }
    return training_dict

def train_loop(countries, args, params):
    country_opts = countries + ['all']
    print('Generating data loaders...')
    data_loaders = getDataLoaders(countries, args.use_graph, args.vec_feature_path,
                                    params['batch_size'])
    for train in country_opts:
        print('\nTraining on {}...'.format(train))
        this_train = data_loaders[train]['train']
        this_val = data_loaders[train]['val']
        # params['run_name'] = 'run0/train-{}'.format(train)
        writer_dir = os.path.join(args.model_dir, 'tb', train)
        writer = SummaryWriter(writer_dir)
        training_dict = loadModels(args, params)
        models = train(training_dict, loss_fns, this_train, this_val, writer, params)

        print('Model trained in {} results:'.format(train))
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
            print('\tValidated in {}'.format(val))
            print('\tSeparate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['imr'], corrs['mated']))

def main():
    args, params = parseArgs()
    # training_dicts = loadModels(args, params)
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    train_loop(countries, args, params)


if __name__ == '__main__':
    main()
