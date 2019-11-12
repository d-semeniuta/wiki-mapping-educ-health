import os
import random
import datetime

from model import GUFNet
from data import GUFAfricaDataset

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
import torch.utils.data as data

from tqdm import tqdm
from tensorboardX import SummaryWriter

def train_model(params, train_countries, test_countries, writer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mated_model = GUFNet('mated', params).to(device)
    imr_model = GUFNet('imr', params).to(device)

    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    mated_optimizer = optim.Adam(
        mated_model.parameters(),
        lr=params['lr'],
    )
    imr_optimizer = optim.Adam(
        imr_model.parameters(),
        lr=params['lr'],
    )

    mated_model.train()
    imr_model.train()
    batch_size = params['batch_size']

    if train_countries == test_countries:
        dataset = GUFAfricaDataset(countries=train_countries)
        validation_split = 0.2
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = data.SubsetRandomSampler(train_indices)
        valid_sampler = data.SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=valid_sampler)
    else:
        train_data = GUFAfricaDataset(countries=train_countries)
        test_data = GUFAfricaDataset(countries=test_countries)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    num_epochs = 100
    loss_fn = nn.MSELoss()
    step = 0
    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        with tqdm(total=len(train_loader)) as progress_bar:
            for i, batch in enumerate(train_loader):
                images, imr, ed_score = batch['image'].to(device), batch['imr'].to(device), batch['ed_score'].to(device)

                mated_out = mated_model.forward(images)
                imr_out = imr_model.forward(images)

                mated_loss = loss_fn(mated_out.squeeze(), ed_score)
                mated_optimizer.zero_grad()
                mated_loss.backward()
                mated_optimizer.step()

                imr_loss = loss_fn(imr_out.squeeze(), imr)
                imr_optimizer.zero_grad()
                imr_loss.backward()
                imr_optimizer.step()

                step += len(batch)
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         MLE_mated=mated_loss.item(),
                                         MLE_imr=imr_loss.item())
                writer.add_scalar('train/MatEd/MLE', mated_loss.item(), step)
                writer.add_scalar('train/IMR/MLE', imr_loss.item(), step)
        # evaluate at end of epoch
        evaluate_model(mated_model, imr_model, test_loader, writer, step)
        imr_model.train()
        mated_model.train()

    return mated_model, imr_model, train_loader

def evaluate_model(mated_model, imr_model, test_loader, writer, step):
    imr_model.eval()
    mated_model.eval()
    imr_ins, imr_outs = [], []
    ed_score_ins, ed_score_outs = [], []
    for batch in test_loader:
        images, imr, ed_score = batch['image'], batch['imr'], batch['ed_score']
        imr_out = imr_model.forward(images)
        ed_score_out = mated_model.forward(images)
        imr_ins.append(imr)
        imr_outs.append(imr_out.detach().squeeze())
        ed_score_ins.append(ed_score)
        ed_score_outs.append(ed_score_out.detach().squeeze())

    imr_ins = torch.cat(imr_ins).numpy()
    imr_outs = torch.cat(imr_outs).numpy()
    ed_score_ins = torch.cat(ed_score_ins).numpy()
    ed_score_outs = torch.cat(ed_score_outs).numpy()
    imr_corr = pearsonr(imr_ins, imr_outs)[0]
    mated_corr = pearsonr(ed_score_ins, ed_score_outs)[0]
    writer.add_scalar('val/IMR/r2', imr_corr, step)
    writer.add_scalar('val/MatEd/r2', mated_corr, step)


def eval_overfit(mated_model, imr_model, train_loader):
    imr_model.eval()
    mated_model.eval()
    for batch in train_loader:
        images, imr, ed_score = batch['image'], batch['imr'], batch['ed_score']
        imr_out = imr_model.forward(images)
        mated_out = mated_model.forward(images)
        for i, o in zip(imr, imr_out):
            print('IMR\tTrue: {}\tPred: {}'.format(i, o.item()))
        imr_corr = pearsonr(imr_out.detach().squeeze().numpy(), imr.numpy())
        print('IMR Corr: {}'.format(imr_corr))
        for i, o in zip(ed_score, mated_out):
            print('Mated\tTrue: {}\tPred: {}'.format(i, o.item()))
        mated_corr = pearsonr(mated_out.detach().squeeze().numpy(), ed_score.numpy())
        print('Mat Ed Corr: {}'.format(mated_corr))


def overfit():
    learning_rates = [3e-4, 1e-3, 5e-3, 1e-2]
    sigmoid_outs = [False, True]
    conv_activations = ["relu", "sigmoid", "none"]
    params = {
        'lr': 1e-3,
        'batch_size': 16,
        'sigmoid_out': False,
        'conv_activation': 'relu'
    }
    exp_name = 'overfit'
    date_time = datetime.datetime.now().strftime("%b-%d-%Y__%H-%M-%S")
    for lr in learning_rates:
        for batch_size in [8]:
            for sigmoid_out in sigmoid_outs:
                for conv_activation in conv_activations:
                    params = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'sigmoid_out': sigmoid_out,
                        'conv_activation': conv_activation
                    }
                    writer = SummaryWriter('runs/{}__{}'.format(exp_name, date_time))
                    mated_model, imr_model, train_loader = train_model(params, 'Ghana', 'Ghana', writer)
                    print(params)
                    eval_overfit(mated_model, imr_model, train_loader)
                    print()


def main():
    params = {
        'lr': 3e-4,
        'batch_size': 16,
        'sigmoid_out': False,
        'conv_activation': 'relu'
    }
    exp_name = '2_more_conv'
    date_time = datetime.datetime.now().strftime("%b-%d-%Y__%H-%M-%S")
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    country_opts = countries + ['all']
    for train in country_opts:
        for test in country_opts:
            print('Training on {}, testing on {}'.format(train, test))
            writer = SummaryWriter('runs/{}__train-{}_test-{}__{}'.format(exp_name, train, test, date_time))
            this_train = countries if train == 'all' else train
            this_test = countries if test == 'all' else test
            train_model(params, this_train, this_test, writer)

if __name__ == '__main__':
    # overfit()
    main()

