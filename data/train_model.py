import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import util_data
import os
import datetime
import time
import pickle
import copy

from models.models import WikiEmbRegressor
from args import get_args

def main():
    full_start = time.time()
    args = get_args()
    USE_GPU = True

    dtype = torch.float32

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logging
    date_time = datetime.datetime.now().strftime("%b-%d-%Y__%H-%M-%S")
    if args.load:
        logdir = os.path.join('runs', args.load_dir)
    else:
        experiment_name = args.exp_name + '_' + date_time
        logdir = os.path.join('runs', experiment_name)
        os.mkdir(logdir)

    logdir_full = os.path.join(os.getcwd(), logdir)
    all_hyperparam_dir = os.path.join(logdir, 'hyperpars')
    hp_writer = SummaryWriter(log_dir=all_hyperparam_dir)

    article_embeddings_dir = os.path.join(os.curdir, 'raw', 'wikipedia', 'doc2vec_embeddings')
    cluster_article_rank_dist_path = os.path.join(os.curdir, 'processed',
                                                       'ClusterNearestArticles_Rank_Dist.csv')

    # hyperparameters
    hps = {}

    tasks = ['MatEd', 'IMR']

    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']#, 'Rwanda']
    cross_countries = False
    if cross_countries:
        countries_train_set, countries_test_sets = util_data.make_country_cross_comparison(countries)
        countries_val_sets = countries_train_set
    else:
        countries_train_set = [['Kenya', 'Ghana', 'Zimbabwe']]
        countries_val_sets = [['Egypt', 'Rwanda']] * len(tasks)
        countries_test_sets = [[['Egypt', 'Rwanda'], ['Zimbabwe']]] * len(tasks)

    # arguments
    smokescreen = args.smokescreen
    # smokescreen = True
    # evaluate_model = True
    if smokescreen:
        epochs = 2
        eval_every = 2
        # evaluate_model = False
        save_every = 2
        batch_sizes = {phase: size for phase, size in zip(['train', 'val', 'test'], [256, 1024, 1024])}
        # eval_batchsize=1000
        countries_train_set = [['Rwanda']]
        countries_val_sets = [['Zimbabwe']]*len(tasks)
        countries_test_sets = [[['Rwanda'], ['Zimbabwe'], ['Benin']]]*len(tasks)
    else:
        epochs = 25
        eval_every = 10
        save_every = 10
        batch_sizes = {phase: size for phase, size in zip(['train', 'val', 'test'], [32, 1024, 1024])}
        # eval_batchsize = 1000
        countries_train_set = [['Rwanda']]
        countries_val_sets = [['Zimbabwe']] * len(tasks)
        countries_test_sets = [[['Rwanda'], ['Zimbabwe'], ['Benin']]] * len(tasks)
    num_workers = 4

    n_articles_nums = [1]#, 3, 5]  #, 7, 10, 15]
    learning_rates = [0.003]
    regularization_strengths = [0]

    best_hp = {}
    best_hp_overall = {}
    best_val_metric_overall = -np.inf  # metric increasing with model performance

    train_loss_histories = []
    train_metric_histories = []
    val_loss_histories = []
    val_metric_histories = []
    train_iter_histories = []
    val_iter_histories = []

    hps = []
    # for i, model in enumerate(models):
    for task in tasks:
        for reg in regularization_strengths:
            for lr in learning_rates:
                for n_articles in n_articles_nums:
                    for countries_train in countries_train_set:
                        hps.append(({'n_articles':n_articles, 'task': task, 'reg':reg, 'lr':lr, 'cty_trn': countries_train}))

    # either load hyperparameters from previous experiment or save current hyperparameters for new experiment
    if args.load:
        with open(os.path.join(all_hyperparam_dir, 'hyperparam_list.pkl'), 'rb') as f:
            hps = pickle.load(f)
    else:
        with open(os.path.join(all_hyperparam_dir, 'hyperparam_list.pkl'), 'wb') as f:
            pickle.dump(hps, f)

    final_loss_metric_dict = {}

    for hp_i, hp in enumerate(hps):
        print('running with ')
        print(hp)

        hp_folder = util_data.make_hp_dir(hp)
        hp_log_dir = os.path.join(logdir, hp_folder)
        model_folder = 'model'
        vis_folder = 'vis'

        assert len(os.path.abspath(hp_log_dir)) < 200, 'subdirectory for specific hyperparameter log too long (over 200 characters, full 260 on Windows), run will not execute properly'

        writer = SummaryWriter(log_dir=hp_log_dir)  # can't use directory inside log_dir for additional writers for some reason

        if not args.load:
            # os.mkdir(hp_log_dir)
            os.mkdir(os.path.join(hp_log_dir, model_folder))
            os.mkdir(os.path.join(hp_log_dir, vis_folder))

        task = hp['task']
        n_articles = hp['n_articles']
        modes = [task] + ['embeddings'] + ['distances']
        country_subsets = {}
        country_subsets['train'] = hp['cty_trn']
        country_subsets['val'] = countries_val_sets[hp_i]

        hp_key = []
        for k , v in hp.items():
            if isinstance(v, list):
                hp_key.append((k, tuple(v)))
            else:
                hp_key.append((k, v))
        hp_key = tuple(hp_key)

        final_loss_metric_dict[hp_key] = {}
        final_loss_metric_dict[hp_key]['train country set'] = country_subsets['train']
        final_loss_metric_dict[hp_key]['val country set'] = country_subsets['val']
        # let individual items of performance dictionary be stored separately with hyperparameter values included
        final_loss_metric_dict[hp_key]['hp'] = hp

        DHS_path = os.path.join(os.path.curdir, 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv')
        # train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
        # val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
        # test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')
        # set_paths = {phase: path for phase, path in zip(['train', 'val', 'test'], [train_path, val_path, test_path])}

        datasets = {phase: util_data.DHS_Wiki_Dataset(DHS_csv_file=DHS_path,
                                                      emb_root_dir=article_embeddings_dir,
                                                      cluster_rank_csv_path=cluster_article_rank_dist_path,
                                                      emb_dim=300, n_articles=n_articles, include_dists=True,
                                                      country_subset=None, task=task,
                                                      transforms=None)
                    for phase in ['train', 'val']}

        datasets['train'].subset(country_subsets['train'])
        datasets['val'].subset(country_subsets['val'])

        data_loaders = {phase: DataLoader(datasets[phase], batch_size=batch_sizes[phase], shuffle=True, num_workers=num_workers)
                        for phase in ['train', 'val']}

        if task == 'MatEd':
            MEL_IMR = True
        elif task == 'IMR':
            MEL_IMR = False
        else:
            assert False, 'Must choose between MatEd and IMR for prediction task for now'

        if args.load:
            model = torch.load(os.path.join(hp_log_dir, model_folder, 'model.pt'))
        else:
            model = WikiEmbRegressor(emb_size=300, n_embs=n_articles, ave_embs=False, concat=True, MEL_IMR=MEL_IMR)

        if not args.eval_only:
            optimizer = torch.optim.Adam(model.parameters(), lr = hp['lr'], weight_decay=hp['reg'])

        model = model.to(device=device)  # move the model parameters to CPU/GPU
        best_model = None

        # track model loss across training
        train_loss_history = []
        val_loss_history = []

        # track model performance metrics across training
        # train_metric_history is evaluated at the same time as val_metric_history
        train_metric_history = []
        val_metric_history = []

        # track iterations of evaluation points across training
        train_iter_history = []
        val_iter_history = []

        best_val_loss = np.inf
        best_val_metric = -np.inf
        t_batches = 0
        epoch_times = []
        iteration_times = []

        if not args.eval_only:
            for e in range(epochs):
                print('Epoch {}/{}'.format(e, epochs))
                epoch_start_time = time.time()
                for i_batch, batch in enumerate(data_loaders['train']):
                    iter_start_time = time.time()
                    x, y = batch['x'], batch['y']
                    x = x.to(device)
                    y = y.to(device)

                    model.train()  # put model to training mode
                    pred = model(x)

                    if task == 'IMR':
                        loss = F.mse_loss(pred, y)
                    if task == 'MatEd':
                        # KL divergence loss, works for continuous distributions (equivalent to cross-entropy)
                        # loss = F.kl_div(pred, y)
                        loss = F.mse_loss(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss_history.append(loss.detach().item())
                    writer.add_scalar('Train Loss', loss.detach().item(), t_batches)
                    train_iter_history.append(t_batches)

                    if t_batches % eval_every == 0:
                        if task == 'MatEd':
                            cont_ed_pred = np.dot(pred.detach().numpy(), np.array([0.0, 1.0, 2.0, 3.0]))
                            cont_ed_y = np.dot(y.detach().numpy(), np.array([0.0, 1.0, 2.0, 3.0]))
                            metric = r2_score(cont_ed_y, cont_ed_pred)
                        else:
                            metric = r2_score(y.detach(), pred.detach())
                        train_metric_history.append(metric)
                        writer.add_scalar('Train metric', metric, t_batches)

                        val_losses = []
                        val_metrics= []
                        model.eval()
                        # EVALUATE ON A SUBSET OF THE VAL SET (WITH SOME MAXIMUM SIZE, OTHERWISE USE ENTIRE VAL SET) DURING TRAINING...
                        # EVALUATE ON FULL VAL SET AFTER TRAINING
                        with torch.no_grad():
                            for i_batch_val, batch_val in enumerate(data_loaders['val']):
                                x, y = batch_val['x'], batch_val['y']
                                x = x.to(device)
                                y = y.to(device)

                                pred = model(x)
                                val_loss = F.mse_loss(pred, y).detach()
                                if task == 'MatEd':
                                    cont_ed_val_pred = np.dot(pred.numpy(), np.array([0.0, 1.0, 2.0, 3.0]))
                                    cont_ed_val_y = np.dot(y.numpy(), np.array([0.0, 1.0, 2.0, 3.0]))
                                    val_metric = r2_score(cont_ed_val_y, cont_ed_val_pred)
                                else:
                                    val_metric = r2_score(y.detach(), pred.detach())
                                val_losses.append(val_loss.item())
                                val_metrics.append(val_metric.item())

                        val_loss_mean = np.mean(np.array(val_losses))
                        val_metric_mean = np.mean(np.array(val_metrics))
                        writer.add_scalar('Validation Loss', val_loss_mean, t_batches)
                        writer.add_scalar('Validation R-Squared', val_metric_mean, t_batches)
                        if val_metric > best_val_metric:
                            # may need to change over to periodic checkpointing without early stopping if models
                            # take more memory and keeping copies around is too expensive
                            best_model = copy.deepcopy(model)
                            best_hp = hp
                            best_val_loss = val_loss_mean
                            best_val_metric = val_metric_mean
                        val_loss_history.append(val_loss_mean)
                        val_metric_history.append(val_metric_mean)
                        val_iter_history.append(t_batches)

                        print('Epoch %d, Iteration %d, loss = %.4f' % (e, t_batches, loss.item()))
                        print('Val loss {:.3f}     Val metric: {:.3f}'.format(val_loss, val_metric))

                    if t_batches % save_every == 0:
                        torch.save(best_model, os.path.join(hp_log_dir, model_folder, 'model.pt'))

                    iter_end_time = time.time()
                    iteration_times.append(iter_end_time-iter_start_time)
                    print('\tIteration runtime (s): {:.2f}    '
                          'Mean iteration runtime (s): {:.2f}'.format(iteration_times[-1],
                                                                      np.mean(np.array(iteration_times))))

                    t_batches += 1

                epoch_end_time = time.time()
                epoch_times.append(epoch_end_time-epoch_start_time)
                print('Epoch runtime (s): {:.2f}'.format(epoch_times[-1]))
                print()

        # save once more after training
        torch.save(best_model, os.path.join(hp_log_dir, model_folder, 'model.pt'))

        train_loss_histories.append(train_loss_history)
        train_metric_histories.append(train_metric_history)
        val_loss_histories.append(val_loss_history)
        val_metric_histories.append(val_metric_history)
        train_iter_histories.append(train_iter_history)
        val_iter_histories.append(val_iter_history)

        # can't store hyperparameter values of strings... might want this later,
        # gives easy comparison of hyperparameters in Tensorboard
        # hp_writer.add_hparams(hp, {'Best Val Loss': val_loss_history[-1], 'Best Score': val_metric_history[-1]})

        if best_val_metric > best_val_metric_overall:
            best_hp_overall = best_hp
            best_val_metric_overall = best_val_metric

        # final evaluation with best (early stopped) model for given set of hyperparameters
        eval_losses = []
        eval_metrics = []
        model.eval()

        if args.eval_only:
            best_model = model

        countries_test_set = countries_test_sets[hp_i]

        print('Evaluating hyperparameters {}'.format(hp))
        phase = 'test' if args.eval_test else 'val'

        # eval_path = os.path.join(os.path.curdir, 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv')
        # eval_dataset = util_data.DHS_Wiki_Dataset(DHS_csv_file=eval_path,
        #                                           emb_root_dir=article_embeddings_dir,
        #                                           cluster_rank_csv_path=cluster_article_rank_dist_path,
        #                                           emb_dim=300, n_articles=n_articles, include_dists=True,
        #                                           country_subset=country_train, task=task,
        #                                           transforms=None)
        # eval_data_loader = DataLoader(eval_dataset, batch_size=100, shuffle=True, num_workers=num_workers)

        for test_country_set in countries_test_sets[hp_i]:
            with torch.no_grad():
                datasets['val'].subset(test_country_set)
                eval_data_loader = DataLoader(datasets['val'], batch_size=250, shuffle=False, num_workers=num_workers)

                for i_batch_val, batch_val in enumerate(eval_data_loader):
                    x, y = batch_val['x'], batch_val['y']
                    x = x.to(device)
                    y = y.to(device)

                    pred = best_model(x)
                    eval_loss = F.mse_loss(pred, y).detach()
                    if task == 'MatEd':
                        cont_ed_val_pred = np.dot(pred.detach().numpy(), np.array([0.0, 1.0, 2.0, 3.0]))
                        cont_ed_val_y = np.dot(y.detach().numpy(), np.array([0.0, 1.0, 2.0, 3.0]))
                        eval_metric = r2_score(cont_ed_val_y, cont_ed_val_pred)
                    else:
                        eval_metric = r2_score(y.detach(), pred.detach())
                    eval_losses.append(eval_loss.item())
                    eval_metrics.append(eval_metric.item())

            eval_loss_mean = np.mean(np.array(eval_losses))
            eval_metric_mean = np.mean(np.array(eval_metrics))
            writer.add_scalar('{} set loss final'.format(phase), eval_loss_mean)
            writer.add_scalar('{} set R-squared final'.format(phase), eval_metric_mean)

            test_country_set_key = tuple(test_country_set)
            final_loss_metric_dict[hp_key][test_country_set_key] = {}
            final_loss_metric_dict[hp_key][test_country_set_key]['loss'] = eval_loss_mean
            final_loss_metric_dict[hp_key][test_country_set_key]['metric'] = eval_metric_mean
            print('Best evaluation loss on {} trained on {}: {}'.format(
                                ', '.join(test_country_set), country_subsets['train'], eval_loss_mean))
            print('Best evaluation metric on {} trained on {}: {}'.format(
                                ', '.join(test_country_set), country_subsets['train'], eval_metric_mean))
            print()

            writer.close()

        with open(os.path.join(hp_log_dir, 'hp_performance_dict.pkl'), 'wb') as f:
            pickle.dump(final_loss_metric_dict[hp_key], f)
        with open(os.path.join(all_hyperparam_dir, 'all_hps_performance_dict.pkl'), 'wb') as f:
            pickle.dump(final_loss_metric_dict, f)


    # save and print best set of hyperparameters if run included training
    if not args.eval_only:
        print('Best hyperparameters overall:')
        print(best_hp_overall)
        print('Best validation score overall: R^2 {}'.format(best_val_metric_overall))
        with open(os.path.join(all_hyperparam_dir, 'best_hyperparam_dict.pkl'), 'wb') as f:
            pickle.dump(best_hp_overall, f)

        loss_fname = 'losses.png'
        metric_fname = 'metrics.png'
        for t, hp in enumerate(hps):
            plt.figure(t)
            plt.title('{}'.format(hp))
            plt.subplot(1, 2, 1)
            plt.plot(np.array(train_iter_histories[t]), train_loss_histories[t],
                               np.array(val_iter_histories[t]), val_loss_histories[t], 'o')
            plt.legend(['Train', 'Validation'])
            plt.xlabel('Iterations')
            plt.ylabel('Loss')

            plt.savefig(os.path.join(logdir_full, util_data.make_hp_dir(hp), vis_folder, loss_fname))

            plt.subplot(1, 2, 2)
            plt.plot(np.array(val_iter_histories[t]), train_metric_histories[t], '-o', label='Training Score: R-squared')
            plt.plot(np.array(val_iter_histories[t]), val_metric_histories[t], '-o', label='Validation Score: R-squared')
            plt.legend(['Train', 'Validation'])
            plt.xlabel('Iterations')
            plt.ylabel('R-squared')

            plt.savefig(os.path.join(logdir_full, util_data.make_hp_dir(hp), vis_folder, metric_fname))
            plt.gcf().set_size_inches(15, 5)

        # Print out training results.
        for t, hp in enumerate(hps):
            print('{} train score: %f val score: %f'.format(hp) % (
                        train_metric_histories[t][-1], val_metric_histories[t][-1]))

        print('best validation performance achieved during cross-validation: %f' % best_val_metric)
        print('with hyperparameters {}'.format(best_hp_overall))

    hp_writer.close()

    full_time = time.time() - full_start
    hours = full_time//3600
    minutes = (full_time - hours*3600.0)//60
    seconds = full_time - hours*3600.0 - minutes*60.0
    print('Total time for all training loops and evaluation: {} hr {} min {} sec'.format(hours, minutes, seconds))

    if smokescreen:
        plt.show()

if __name__ == '__main__':
    main()

    # TODO
    # 1. DONE switch to pytorch dataloader, test data being loaded correctly,
    #           look into optimization included in out-of-the-box dataloader, may not want to load items individually
    #           generalize for multi-modal data streams
    # 1.5 NEED data preloading, preload all DHS .csv files and article embeddings before hyperparameter loop,
    #           then pass embeddings between data loaders
    # 2. DONE, NOT TESTED BUT SHOULD BE GOOD get full pytorch tensorboardx data logging
    # 2.5 add verbosity levels (allow for printing IO examples of the data with predictions) and full names to various metrics and models, generalize for multiple metrics
    # 3. DONE add final model evaluation at the end of training over entire evaluation/test set depending on selected options


    #           CHECKED tensorboard outputting
    #           country-wise evaluation:
    #                   DONE train and test across countries and within all countries, train also on all countries
    #                   add leave-one-country-out cross-validation likely next week, if models train quickly, might be able to do this tonight
    #
    #                   test on each country test set for a given country train set, save result to both a pickle file
    #                   with just the abbreviated country names and to a tensorboard hyperparameter entry
    #
    #                   have separate train and val country sets

    # 3.5 DONE model checkpointing, model loading for both continuation of training and for evaluation only with and without test set
    #           include hyperparameter/parser args saving as well
    # 4. DONE PARTLY parser args, include model saving option that can be turned off for testing
    # 5. record/print runtimes
    # 6. NEED get running on the VM with GPU
    # 7. data preprocessing, maybe augmentation, check into overfitting on larger data sets
    # 8. DONE add maternal education predictions
    # 9. NEED country-wise cross-validation
    # 10. NEED to speed up the train loop? might need to memory map the data set, look into directory structures for
    #           fast accessing, preloading

    # TODO: EXPERIMENTS
    # 1. overfit on some smaller countries
    # 2. rerun experiments with full country set


    # NEEDS TO BE CHECKED load data
    # DONE need to design train/val/test split now, keep clean
    # CURRENTLY COUNTRY BALANCED, NEEDS TO BE BALANCED OVER DISTRIBUTION OF CONTINUOUS VALUES? (need to decide on data sets... should choose class balanced and geo-distributed splits, make them and then stick with them)
    # DONE original paper didn't keep a clean train/val/test split, just used different countries to test generalization
    # DONE do the same thing, but try to choose country splits such that train and test countries don't border each other

    # DONE need to decide on how to subset the data, want to subset by country, want to train and test on all countries in set
    # DONE but seems like that will lead to leakage across data set split. That's fine for within country, that's what
    # DONE happened in the original paper, just keep it, rely on out-of-country testing for generalization properties

    # DONE iterate over train sets with multiple val sets? or just get trained models then evaluate on specific val sets
    # DONE that would be simplest, break up training and testing beyond val loss over training process
    # DONE make a util function for producing country cross training (evaluate on validation set, not test set)

    # could preload article embeddings particularly if there are only ~50000 geolocated articles in Afrcia. Just preload
    # all articles in the set of K nearest neighbors


    # ALL DONE first design data structure for all data sets: CSV of IO pairs, import with pandas data frame, then convert to
    # dictionary of numpy arrays? That'd be simpler, I don't want to deal with Pandas, but might be better once
    # I understand it better, will need to convert to PyTorch Tensors eventually, might be easier to standardize around
    # numpy than pandas, how easy is it to convert from one to the other?
    # epochs of shuffled data or batches of random samples with replacement from data generator?
    # shuffling data sets (indices?) can take time, could just maintain list of remaining indices over epoch, remove
    # indices as they're used, and sample indices into list of remaining data point indices from shrinking number
    # of remaining data set size
    # NEED TO TRANSFER TO PYTORCH DATALOADER

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
    # DONE combinatorial grid search? Or just complete hyperparameter tuples? Option for either should be given
    # LATER might also want random hyperparameter search... hyperopt might be too much though
    # want logging that allows for compartmental additions, new variables to be logged on the fly, then same logger can
    # be used in training and analysis if desired

    # test/analyze model:
    # LATER would be nice to have statistical testing on testing, STD, confidence intervals,
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