import torch
import torch.nn.functional as F
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import util_data
import os
import datetime
from models.models import WikiEmbRegressor


USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

# logging
experiment_name = 'test'
date_time = datetime.datetime.now().strftime("%b-%d-%Y__%H:%M:%S")
logdir = os.path.join(os.curdir, 'runs')
model_folder = 'model'
vis_folder = 'vis'

# writer = SummaryWriter(os.path.join(logdir, experiment_name + date_time))


# hyperparameters
hps = {}

countries = [['Ghana'], ['Zimbabwe'], ['Kenya'], ['Egypt'], ['Rwanda']]
cross_countries = False
if cross_countries:
    countries_train_set, countries_test_set = util_data.make_country_cross_comparison(countries)
else:
    countries_train_set = [['Kenya', 'Ghana', 'Zimbabwe']]
    countries_test_set = [['Egypt', 'Rwanda']]

# arguments
smokescreen=True
if smokescreen:
    epochs = 2
    print_every = 5
    save_every = 5
    batchsize=256
    eval_batchsize = 10
    countries_train_set = [['Rwanda']]
    countries_test_set = [['Rwanda']]
else:
    epochs = 10
    print_every = 50
    save_every = 100
    batchsize = 32
    eval_batchsize = 100

n_articles_nums = [1, 3, 5]  #, 7, 10, 15]
learning_rates = [0.003]
regularization_strengths = [0]
tasks = ['IMR'] #, 'MatEd']
batchsize = 32

train_loss_histories = []
train_metric_histories = []
val_loss_histories = []
val_metric_histories = []

hps = []
# for i, model in enumerate(models):
for reg in regularization_strengths:
    for lr in learning_rates:
        for n_articles in n_articles_nums:
            for task in tasks:
                for countries_train in countries_train_set:
                    hps.append(({'n_articles':n_articles, 'task': task, 'reg':reg, 'lr':lr, 'countries_train': countries_train}))

for hp in hps:
    print('running with ')
    print(hp)

    e = 0
    task = hp['task']
    n_articles = hp['n_articles']
    modes = [task] + ['embeddings'] + ['distances']
    country_train = hp['countries_train']

    train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
    val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
    test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

    data_loader = util_data.DataLoader(modes, train_path, val_path, test_path,
                             country_subset=country_train,
                             args=None, preload=True, seed=0, K=n_articles)
    if task == 'MatEd':
        MEL_IMR = True
    elif task == 'IMR':
        MEL_IMR = False
    else:
        assert False, 'Must choose between MatEd and IMR for prediction task for now'

    model = WikiEmbRegressor(emb_size=300, n_embs=n_articles, ave_embs=False, concat=True, MEL_IMR=MEL_IMR)
    optimizer = torch.optim.Adam(model.parameters(), lr = hp['lr'], weight_decay=hp['reg'])

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    best_model = None
    train_loss_history = []
    val_loss_history = []
    train_metric_history = []
    val_metric_history = []

    # train_acc_history = []
    # val_acc_history = []
    best_val_loss = 0.0
    best_val_metric = -1.0
    t_batches = 0
    while e < epochs:
        print('Epoch: {}'.format(e))
        t_batches += 1

        x, y = data_loader.sample_batch(batchsize=batchsize, batch_type='train')
        model.train()  # put model to training mode
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)

        pred = model(x)

        if task == 'IMR':
            loss = F.mse_loss(pred, y)
        if task == 'MatEd':
            loss = F.mse_loss(pred, y)
        # loss = F.cross_entropy(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())

        if t_batches % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t_batches, loss.item()))
            # train_acc = check_accuracy(loader_train, model)
            # val_acc = check_accuracy(loader_val, model)
            train_metric_history = r2_score(y, pred)

            x, y = data_loader.sample_batch(batchsize=eval_batchsize, batch_type='val')
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            pred = model(x)
            val_loss = F.mse_loss(pred, y)
            val_loss_history.append(val_loss)
            val_metric = r2_score(y, pred)
            val_metric_history.append(val_metric)
            print('Val loss {}     Val metric: {}'.format(val_loss, val_metric))

            # train_acc_history.append(train_acc)
            # val_acc_history.append(val_acc)
            print()
            if val_loss > best_val_loss:
                # best_model = model
                best_hp = hp
                best_val_loss = val_loss
                best_val_metric = val_metric

        if t_batches % save_every == 0:
            # torch.save(model, model_save_path)
            pass

        if data_loader.epoch_flag:
            e += 1
            data_loader.epoch_flag = False

    # best_models.append(best_model)
    train_loss_histories.append(train_loss_history)
    train_metric_histories.append(train_metric_history)
    val_loss_histories.append(val_loss_history)
    val_metric_histories.append(val_metric_history)

    if best_val_metric > best_val_metric_overall:
        best_hp_overall = best_hp
        # best_model_overall = best_model
        best_val_metric_overall = best_val_metric

# print(best_model_overall)
print(best_hp_overall)
print('Best validation score overall: R^2 {}'.format(best_val_metric_overall))

for t, hp in enumerate(hps):
    plt.figure(t)
    plt.title('{}'.format(hp))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_loss_histories[t]), train_loss_histories[t],
                       len(val_loss_histories[t]), val_loss_histories[t], 'o'))
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(logdir, vis_folder, str(hp) + '_loss'))
    # plt.title('MSE loss over training')

    plt.subplot(1, 2, 2)
    plt.plot(train_metric_histories[t], '-o', label='Training Score: R-squared')
    plt.plot(val_metric_histories[t], '-o', label='Validation Score: R-squared')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Iterations')
    plt.ylabel('R-squared')
    plt.savefig(os.path.join(logdir, vis_folder, str(hp) + '_metric'))

plt.gcf().set_size_inches(15, 5)
plt.show()

# Print out results.
for t, hp in enumerate(hps):
    print('{} train score: %f val score: %f'.format(hp) % (
                train_metric_histories[t][-1], val_metric_histories[t][-1]))

print('best validation accuracy achieved during cross-validation: %f' % best_val_metric)
print('with hyperparameters {}'.format(best_hp_overall))

# model = best_model_overall

# reinitialize best model for final training
# model.apply(init_weights)

# model = models[0]
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.5e-3)


if __name__ == '__main__':
    pass
    # load data
    # need to design train/val/test split now, keep clean
    # (need to decide on data sets... should choose class balanced and geo-distributed splits, make them and then stick with them)
    # original paper didn't keep a clean train/val/test split, just used different countries to test generalization

    # do the same thing, but try to choose country splits such that train and test countries don't border each other

    # need to decide on how to subset the data, want to subset by country, want to train and test on all countries in set
    # but seems like that will lead to leakage across data set split. That's fine for within country, that's what
    # happened in the original paper, just keep it, rely on out-of-country testing for generalization properties

    # iterate over train sets with multiple val sets? or just get trained models then evaluate on specific val sets
    # that would be simplest, break up training and testing beyond val loss over training process
    # make a util function for producing country cross training (evaluate on validation set, not test set)

    # could preload article embeddings particularly if there are only ~50000 geolocated articles in Afrcia. Just preload
    # all articles in the set of K nearest neighbors


    # first design data structure for all data sets: CSV of IO pairs, import with pandas data frame, then convert to
    # dictionary of numpy arrays? That'd be simpler, I don't want to deal with Pandas, but might be better once
    # I understand it better, will need to convert to PyTorch Tensors eventually, might be easier to standardize around
    # numpy than pandas, how easy is it to convert from one to the other?
    # epochs of shuffled data or batches of random samples with replacement from data generator?
    # shuffling data sets (indices?) can take time, could just maintain list of remaining indices over epoch, remove
    # indices as they're used, and sample indices into list of remaining data point indices from shrinking number
    # of remaining data set size

    # data normalization

    # data augmentation?


    # ideally would load data into memory if possible, need to consider model size (not too big) and data size
    # (potentially not big depending on what is loaded, whether just embeddings, article text, GUF, WikiMedia


    # want list of I/O pairs, allow for multimodal data (on input and output),
    # maybe two dictionaries for I and O, key for every mode of I and O data
    #


    # instantiate model(s), likely needs to be done in the training loop, pass in model object and hyperparameters,
    # unless hyperparameter loop is here or an external function to train...
    # no reason to constrain training to handle HP selection, separate functions >> hyperopt functionality


    # WOULD IDEALLY BE ABLE TO RUN BASELINES AS WELL, ANY MODELS,
    # NEED SOME FORM OF ENCAPSULATION OF MODELS,
    # maybe make a model class with train/test method, links to whatever is needed
    # how to perform compare across models? final performance across models?
    # can write analysis functions, assign whichever ones apply, run them, return results in dictionary

    # define hyperparameters to test


    # train model:
    # combinatorial grid search? Or just complete hyperparameter tuples? Option for either should be given
    # might also want random hyperparameter search... hyperopt might be too much though, not worth it for this week at
    # least - focus on grid search for now, get some good analyses
    # want logging that allows for compartmental additions, new variables to be logged on the fly, then same logger ca
    # be used in training and analysis if desired

    # test/analyze model:
    # would be nice to have statistical testing on testing, STD, confidence intervals,
    # cross-validation using train and val sets with combined statistics, also training with multiple random
    # initializations, also need to control random seed across code as much as possible
    # design visualizations/analyses to run with generic functions, anything specific to the training process, like loss
    # curves, need to be dealt with and returned by that specific training process

    # visualizations:
    # loss/performance curves over training
    # performance versus choices of hyperparameters, specifically for linear grid searches?
    # We could show two-dimensional grid searches as well with a color plot, would need to detect
    # dimensionality of grid search


    # HAVE TEST OPTION

    # save best model (out of all hyperparameter combinations for now), visualizations, analyses

    # code testing:
    # have automated tests for each function, smokescreen,
    # verify files are being saved, verify number of loss points in the training logs, hyperparameter logs?
    #