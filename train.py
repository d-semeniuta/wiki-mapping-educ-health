"""
Train the model
"""

import argparse
import os
import json

import torch

from scipy.stats import pearsonr

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'last'
    args = parser.parse_args()
    param_loc = os.path.join(args.model_dir, 'params.json')
    params = json.load(param_loc)
    params['device'] =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params['model_dir'] = args.model_dir
    return args, params


def train(models, optimizers, loss_fns, train_loader, val_loader, writer, params):
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
    epoch, step = 0, 0
    total_batches = params['num_epochs'] * len(train_loader)
    best_corrs = {'imr': -1, 'mated': -1}
    with tqdm(total=total_batches) as progress_bar:
        epoch += 1
        while epoch != params['num_epochs']:
            for i, batch in enumerate(train_loader):
                step += len(batch)
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
                last_out = os.path.join(params['model_dir'], '{}.last.pth'.format(task))
                dict_to_save = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizers[task].state_dict()
                }
                torch.save(dict_to_save, last_out)
                if corrs[task] > best_corrs[task]:
                    best_corrs[task] = corrs[task]
                    best_out = os.path.join(params['model_dir'], '{}.best.pth'.format(task))
                    torch.save(dict_to_save, best_out)


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




if __name__ == '__main__':
    main()
