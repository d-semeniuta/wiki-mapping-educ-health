import os
import random

from model import GUFNet
from data import GUFAfricaDataset

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data

from tqdm import tqdm
# from tensorboardX import SummaryWriter


def main():
    # tbx = SummaryWriter(os.path.abspath('./tb'))

    model = GUFNet('imr')

    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    optimizer = optim.SGD(
        model.parameters(),
        lr= 1e-3,
    )

    model.train()
    train_countries = ['Ghana']
    train_data = GUFAfricaDataset(countries=train_countries)
    batch_size, subsample_size = 1, 1
    train_data_subset = data.Subset(train_data, range(subsample_size))
    train_loader = data.DataLoader(train_data_subset, batch_size=batch_size, shuffle=True)
    num_epochs = 50
    loss_fn = nn.MSELoss()
    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        with tqdm(total=len(train_loader)) as progress_bar:
            for i, batch in enumerate(train_loader):
                images, imr, ed_score = batch['image'], batch['imr'], batch['ed_score']

                out = model.forward(images)

                loss = loss_fn(out, imr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss)

    model.eval()
    for batch in train_loader:
        images, imr, ed_score = batch['image'], batch['imr'], batch['ed_score']
        out = model.forward(images)
        # sample = train_data_subset[i]
        # img = sample['image']
        # imr = sample['imr']
        # pred = model.forward(img.unsqueeze(1))
        for i, o in zip(imr, out):
            print('True: {}\tPred: {}'.format(i, o.item()))
    # print('end:', loss)
if __name__ == '__main__':
    main()
