import os
import random

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
        validation_split = 0.2
        train_data = GUFAfricaDataset(countries=train_countries)
        train_dataset_size = len(train_data)
        indices = list(range(train_dataset_size))
        split = int(np.floor(validation_split * train_dataset_size))
        np.random.shuffle(indices)
        train_indices = indices[split:]
        train_sampler = data.SubsetRandomSampler(train_indices)
        train_loader = data.DataLoader(train_data, batch_size=batch_size,
                                        sampler=train_sampler)

        test_data = GUFAfricaDataset(countries=test_countries)
        test_dataset_size = len(test_data)
        indices = list(range(test_dataset_size))
        split = int(np.floor(validation_split * test_dataset_size))
        np.random.shuffle(indices)
        val_indices  = indices[:split]
        valid_sampler = data.SubsetRandomSampler(val_indices)
        test_loader = data.DataLoader(test_data, batch_size=batch_size,
                                        sampler=train_sampler)

    num_epochs = params['num_epochs']
    loss_fn = nn.MSELoss()
    step = 0
    epoch = 0
    eval_every = 5
    with tqdm(total=num_epochs) as progress_bar:
        while epoch != num_epochs:
            epoch += 1
            for i, batch in enumerate(train_loader):
                images, imr, ed_score = batch['image'].to(device), batch['imr'].to(device), batch['ed_score'].to(device)

                mated_out = mated_model.forward(images)
                imr_out = imr_model.forward(images)

                mated_loss = loss_fn(mated_out.squeeze(-1), ed_score)
                mated_optimizer.zero_grad()
                mated_loss.backward()
                mated_optimizer.step()

                imr_loss = loss_fn(imr_out.squeeze(-1), imr)
                imr_optimizer.zero_grad()
                imr_loss.backward()
                imr_optimizer.step()

                step += len(batch)
                writer.add_scalar('train/MatEd/MLE', mated_loss.item(), step)
                writer.add_scalar('train/IMR/MLE', imr_loss.item(), step)
            # evaluate at end of 5 epochs
            if epoch % eval_every == 0:
                imr_corr, mated_corr = evaluate_model(mated_model, imr_model, test_loader)
                writer.add_scalar('val/IMR/r2', imr_corr, epoch)
                writer.add_scalar('val/MatEd/r2', mated_corr, epoch)
                imr_model.train()
                mated_model.train()
                progress_bar.update(eval_every)
                progress_bar.set_postfix(epoch=epoch,
                                         MLE_mated=mated_loss.item(),
                                         MLE_imr=imr_loss.item(),
                                         R2_mated=mated_corr,
                                         R2_imr=imr_corr)

    return mated_model, imr_model, train_loader, test_loader

def evaluate_model(mated_model, imr_model, test_loader):
    imr_model.eval()
    mated_model.eval()
    imr_ins, imr_outs = [], []
    ed_score_ins, ed_score_outs = [], []
    for batch in test_loader:
        images, imr, ed_score = batch['image'], batch['imr'], batch['ed_score']
        imr_out = imr_model.forward(images)
        ed_score_out = mated_model.forward(images)
        imr_ins.append(imr)
        imr_outs.append(imr_out.detach().squeeze(-1))
        ed_score_ins.append(ed_score)
        ed_score_outs.append(ed_score_out.detach().squeeze(-1))

    imr_ins = torch.cat(imr_ins).numpy()
    imr_outs = torch.cat(imr_outs).numpy()
    ed_score_ins = torch.cat(ed_score_ins).numpy()
    ed_score_outs = torch.cat(ed_score_outs).numpy()
    imr_corr = pearsonr(imr_ins, imr_outs)[0]
    mated_corr = pearsonr(ed_score_ins, ed_score_outs)[0]
    return imr_corr, mated_corr


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
    learning_rates = [1e-3, 5e-3, 1e-2]
    sigmoid_outs = [False, True]
    conv_activations = ["relu", "sigmoid", "none"]
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
                    mated_model, imr_model, train_loader = train(params, 'Ghana', overfit=True)
                    print(params)
                    eval_overfit(mated_model, imr_model, train_loader)
                    print()


def main():
    params = {
        'lr': 8e-4,
        'batch_size': 16,
        'sigmoid_out': False,
        'conv_activation': 'relu',
        'num_epochs': 150
    }
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    country_opts = countries + ['all']
    for train in country_opts:
        for test in country_opts:
            if train == 'Ghana' and test != 'all':
                continue
            print('Training on {}, testing on {}'.format(train, test))
            writer = SummaryWriter('runs2/train-{}_test-{}'.format(train, test))
            this_train = [country for country in countries if country is not test] if train == 'all' else train
            this_test = [country for country in countries if country is not train] if test == 'all' else test
            mated_model, imr_model, train_loader, test_loader = train_model(params, this_train, this_test, writer)
            imr_corr, mated_corr = evaluate_model(mated_model, imr_model, test_loader)
            print('Final model results:\n\tIMR corr: {}\n\tMatEd corr: {}'.format(imr_corr, mated_corr))
            print()


if __name__ == '__main__':
    main()
