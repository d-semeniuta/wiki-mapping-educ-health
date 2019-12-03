raise RuntimeError('This train loop has been deprecated, if you are sure you want to, comment out the raise')

import os
import random
import pdb

from model import GUFNet
from data import GUFAfricaDataset, getDataLoaders

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr, linregress
import torch.utils.data as data

import plotly.graph_objects as go


from tqdm import tqdm
from tensorboardX import SummaryWriter

def split_dataset(dataset, batch_size=16, validation_split=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = data.SubsetRandomSampler(train_indices)
    valid_sampler = data.SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        sampler=valid_sampler)
    return train_loader, val_loader


def generate_loaders(countries):


    data_loaders = {}
    for country in countries:
        data = GUFAfricaDataset(countries=country)
        others = [c for c in countries if c is not country]
        others_data = GUFAfricaDataset(countries=others)
        data_loaders[country] = {}
        data_loaders[country]['train'], data_loaders[country]['val'] = split_dataset(data)
        data_loaders[country]['others'] = {}
        data_loaders[country]['others']['train'], data_loaders[country]['others']['val'] = split_dataset(others_data)
    alldata = GUFAfricaDataset(countries=countries)
    data_loaders['all'] = {}
    data_loaders['all']['train'], data_loaders['all']['val'] = split_dataset(alldata)

    return data_loaders

def get_loss(out, task, labels, loss_fn):
    imr, ed_score = labels
    if task == 'imr':
        loss = loss_fn(out, imr.unsqueeze(-1))
    elif task == 'mated':
        loss = loss_fn(out, ed_score.unsqueeze(-1))
    else:
        loss = loss_fn(out[0], imr.unsqueeze(-1)) + loss_fn(out[1], ed_score.unsqueeze(-1))
    return loss

def train_model(params, train_loader, val_loader, writer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dict = {
         'imr': {}, 'mated': {}, 'both': {}
    }
    tasks = model_dict.keys()
    for task in model_dict.keys():
        model = GUFNet(task, params).to(device)
        model_dict[task]['optimizer'] = optim.Adam(
            model.parameters(), lr=params['lr']
        )
        model.train()
        model_dict[task]['model'] = model
    batch_size = params['batch_size']

    num_epochs = params['num_epochs']
    loss_fn = nn.MSELoss()
    step = 0
    epoch = 0
    eval_every = params['eval_every']
    save_every = params['save_every']
    plot_dir = './plots/{}'.format(params['run_name'].replace('/', '_'))
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_info = {
        'save_dir' : plot_dir,
        'run_name' : params['run_name']
    }
    checkpoint_dir = os.path.join('./checkpoints', params['run_name'].split('/')[0])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with tqdm(total=num_epochs) as progress_bar:
        while epoch != num_epochs:
            epoch += 1
            for i, batch in enumerate(train_loader):
                images, imr, ed_score = batch['image'].to(device), batch['imr'].to(device), batch['ed_score'].to(device)

                step += len(batch)
                outs = {}
                for task, dict in model_dict.items():
                    out = dict['model'].forward(images)
                    outs[task] = out
                    loss = get_loss(out, task, (imr, ed_score), loss_fn)
                    dict['optimizer'].zero_grad()
                    loss.backward()
                    dict['optimizer'].step()
                    writer.add_scalar('train/{}/MLE'.format(task), loss.item(), step)
            progress_bar.update(1)

            # evaluate at end of eval_every epochs
            if epoch % eval_every == 0:
                models = {task: dict['model'] for (task, dict) in model_dict.items()}
                plot_info['epoch'] = epoch
                corrs, losses = evaluate_model(models, val_loader, loss_fn)
                writer.add_scalar('val/IMR/R2', corrs['imr'], epoch)
                writer.add_scalar('val/MatEd/R2', corrs['mated'], epoch)
                writer.add_scalar('val/IMR_both/R2', corrs['both']['imr'], epoch)
                writer.add_scalar('val/MatEd_both/R2', corrs['both']['mated'], epoch)
                for task, loss in losses.items():
                    writer.add_scalar('val/{}/MLE'.format(task), loss, epoch)
                for model in models.values():
                    model.train()
                # progress_bar.update(eval_every)
                progress_bar.set_postfix(
                    R2_IMR=corrs['imr'],
                    R2_MatEd=corrs['mated'],
                    R2_IMR_both=corrs['both']['imr'],
                    R2_MatEd_both=corrs['both']['mated'],
                    MLE_imr=losses['imr'],
                    MLE_mated=losses['mated'],
                    MLE_both=losses['both']
                )
            if epoch % save_every == 0:
                for task, dict in model_dict.items():
                    model = dict['model']
                    out_file = './checkpoints/{}-{}.last.pth'.format(params['run_name'], task)
                    torch.save(model.state_dict(), out_file)
    return {task: dict['model'] for (task, dict) in model_dict.items()}

def evaluate_model(models, val_loader, loss_fn, plot_preds=False, plot_info=None):
    for model in models.values():
        model.eval()
    ins = {'imr': [], 'mated': []}
    outs = {'imr': [], 'mated': [], 'both': {'imr': [], 'mated': []}}
    for batch in val_loader:
        images, imr, ed_score = batch['image'], batch['imr'], batch['ed_score']
        for task, model in models.items():
            out = model.forward(images)
            if task == 'both':
                outs[task]['imr'].append(out[0].detach().squeeze(-1))
                outs[task]['mated'].append(out[1].detach().squeeze(-1))
            else:
                outs[task].append(out.detach().squeeze(-1))
        ins['imr'].append(imr)
        ins['mated'].append(ed_score)

    ins['imr'] = torch.cat(ins['imr']).numpy()
    ins['mated'] = torch.cat(ins['mated']).numpy()
    corrs = {}
    losses = {}
    cat_outs = {}
    for task in outs.keys():
        if task == 'both':
            corrs[task] = {}
            cat_outs[task] = {}
            losses[task] = 0
            for inner_task in outs[task].keys():
                out = torch.cat(outs[task][inner_task]).numpy()
                cat_outs[task][inner_task] = out
                corrs[task][inner_task] = pearsonr(ins[inner_task], out)[0]
                losses[task] += loss_fn(torch.tensor(ins[inner_task]), torch.tensor(out)).item()
        else:
            out = torch.cat(outs[task]).numpy()
            cat_outs[task] = out
            corrs[task] = pearsonr(ins[task], out)[0]
            losses[task] = loss_fn(torch.tensor(ins[task]), torch.tensor(out)).item()
    if plot_preds:
        if plot_info is None:
            raise(ValueError('Missing plot params'))
        plotPreds(ins, cat_outs, corrs, plot_info)
    return corrs, losses

def plotSingle(ins, outs, corr, save_loc, title, task):
    trace1 = go.Scatter(
        x=ins,
        y=outs,
        mode='markers',
        name='Predictions'
    )
    slope, intercept, r_value, p_value, std_err = linregress(ins,outs)
    max_val = 3 if task == 'mated' else 1
    xi = np.arange(0,max_val,0.01)
    line = slope*xi+intercept

    trace2 = go.Scatter(
        x=xi,
        y=line,
        mode='lines',
        name='Fit'
    )

    annotation = go.layout.Annotation(
        x=0.05,
        y=max_val-0.05,
        text='$R^2 = {:.3f}$'.format(corr),
        showarrow=False,
    )
    layout = go.Layout(
        title = title,
        xaxis_title = 'Ground Truth',
        yaxis_title = 'Predictions',
        annotations=[annotation]
    )

    fig=go.Figure(data=[trace1,trace2], layout=layout)

    fig.write_image(save_loc)

def plotPreds(ins, outs, corrs, plot_info):
    tasks = outs.keys()
    for task in tasks:
        if task == 'both':
            for inner_task in ['imr', 'mated']:
                this_in = ins[inner_task]
                this_out = outs[task][inner_task]
                corr = corrs[task][inner_task]
                save_loc = '{}/{}-{}.png'.format(plot_info['save_dir'], task, inner_task)
                title = '{} {} {}'.format(plot_info['run_name'], task, inner_task)
                plotSingle(this_in, this_out, corr, save_loc, title, inner_task)
        else:
            this_in = ins[task]
            this_out = outs[task]
            corr = corrs[task]
            save_loc = '{}/{}.png'.format(plot_info['save_dir'], task)
            title = '{} {}'.format(plot_info['run_name'], task)
            plotSingle(this_in, this_out, corr, save_loc, title, task)

def eval_overfit(mated_model, imr_model, both_model, train_loader):
    imr_model.eval()
    mated_model.eval()
    both_model.eval()
    for batch in train_loader:
        images, imr, ed_score = batch['image'], batch['imr'], batch['ed_score']
        imr_out = imr_model.forward(images)
        mated_out = mated_model.forward(images)
        imr_both_out, mated_both_out = both_model.forward(images)
        for i, o, b in zip(imr, imr_out, imr_both_out):
            print('IMR\tTrue: {}\tPred: {}\tBoth Pred: {}'.format(i, o.item(), b.item()))
        imr_corr = pearsonr(imr_out.detach().squeeze().numpy(), imr.numpy())
        imr_both_corr = pearsonr(imr_both_out.detach().squeeze().numpy(), imr.numpy())
        print('IMR Corr Single: {}\tBoth: {}'.format(imr_corr, imr_both_corr))
        for i, o, b in zip(ed_score, mated_out, mated_both_out):
            print('Mated\tTrue: {}\tPred: {}Both Pred: {}'.format(i, o.item(), b.item()))
        mated_corr = pearsonr(mated_out.detach().squeeze().numpy(), ed_score.numpy())
        mated_both_corr = pearsonr(mated_both_out.detach().squeeze().numpy(), ed_score.numpy())
        print('MatEd Corr Single: {}\tBoth: {}'.format(mated_corr, mated_both_corr))

def subsample_data_loader(subsample_size=8):
    dataset = GUFAfricaDataset(countries='Ghana')
    train_sampler = data.SubsetRandomSampler(list(range(subsample_size)))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=subsample_size,
                                        sampler=train_sampler)
    return train_loader

def overfit():
    learning_rates = [1e-3, 5e-3, 1e-2]
    sigmoid_outs = [False, True]
    conv_activations = ["relu", "sigmoid", "none"]
    data_loader = subsample_data_loader()
    i = 0
    for lr in learning_rates:
        for batch_size in [8]:
            for sigmoid_out in sigmoid_outs:
                for conv_activation in conv_activations:
                    writer = SummaryWriter('overfitting/param_set_{}'.format(i))
                    params = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'sigmoid_out': sigmoid_out,
                        'conv_activation': conv_activation,
                        'num_epochs': 200
                    }
                    mated_model, imr_model, both_model = train_model(params, data_loader,
                                                                        data_loader, writer)
                    print(params)
                    eval_overfit(mated_model, imr_model, both_model, data_loader)
                    print()
                    i += 1
                    return

def just_Ghana(params):
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    country_opts = countries + ['all']
    print('Generating data loaders...')
    data_loaders = generate_loaders(countries)
    for train in ['Ghana']:
        print('Training on {}'.format(train))
        this_train = data_loaders[train]['train']
        this_val = data_loaders[train]['val']
        params['run_name'] = 'run0/train-{}'.format(train)
        writer = SummaryWriter('tb/{}'.format(params['run_name']))
        # models = train_model(params, this_train, this_val, writer)
        models = {}
        models['both'] = GUFNet('both', params)
        models['both'].load_state_dict(torch.load('checkpoints/run0/train-Ghana-both-epoch-150.pt'))

        print('Model trained in {} results:'.format(train))
        for val in country_opts:
            if val == 'all' and train != 'all':
                val_loader = data_loaders[train]['others']['val']
            else:
                val_loader = data_loaders[val]['val']
            plot_info = {
                'save_dir' : './plots/{}/val-{}'.format(params['run_name'].replace('/', '_'), val),
                'run_name' : '{} val-{}'.format(params['run_name'], val)
            }
            if not os.path.exists(plot_info['save_dir']):
                os.makedirs(plot_info['save_dir'])
            corrs, losses = evaluate_model(models, val_loader, nn.MSELoss(), plot_preds=True, plot_info=plot_info)
            print('\tValidated in {}'.format(val))
            # print('\tSeparate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['imr'], corrs['mated']))
            print('\tBoth Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['both']['imr'], corrs['both']['mated']))

def big_loop(params):
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    country_opts = countries + ['all']
    print('Generating data loaders...')
    data_loaders = generate_loaders(countries)
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    for train in country_opts:
        print('Training on {}'.format(train))
        this_train = data_loaders[train]['train']
        this_val = data_loaders[train]['val']
        params['run_name'] = 'run0/train-{}'.format(train)
        writer = SummaryWriter('tb/{}'.format(params['run_name']))
        models = train_model(params, this_train, this_val, writer)

        print('Model trained in {} results:'.format(train))
        with open('./logs/{}_train.txt'.format(train), 'w') as log_file:
            for val in country_opts:
                if val == 'all' and train != 'all':
                    val_loader = data_loaders[train]['others']['val']
                else:
                    val_loader = data_loaders[val]['val']
                plot_info = {
                    'save_dir' : './plots/{}/val-{}'.format(params['run_name'].replace('/', '_'), val),
                    'run_name' : '{} val-{}'.format(params['run_name'], val)
                }
                if not os.path.exists(plot_info['save_dir']):
                    os.makedirs(plot_info['save_dir'])
                corrs, losses = evaluate_model(models, val_loader, nn.MSELoss(), plot_preds=True, plot_info=plot_info)
                log_file.write("Validated in {}\n".format(val))
                log_file.write("Separate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}\n".format(corrs['imr'], corrs['mated']))
                log_file.write("Both Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}\n".format(corrs['both']['imr'], corrs['both']['mated']))
                print('\tValidated in {}'.format(val))
                print('\tSeparate Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['imr'], corrs['mated']))
                print('\tBoth Model - IMR corr: {:.3f}\t\tMatEd corr: {:.3f}'.format(corrs['both']['imr'], corrs['both']['mated']))

def main():
    params = {
        'lr': 8e-4,
        'batch_size': 16,
        'sigmoid_out': False,
        'conv_activation': 'relu',
        'num_epochs': 120,
        'eval_every': 15,
        'save_every': 30,
        'plot_preds': True
    }
    big_loop(params)
    # overfit()
    # just_Ghana(params)


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
