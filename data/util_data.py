import os
import numpy as np
import pandas as pd
from sklearn import neighbors
import time
import torch
from torch.utils.data import Dataset, DataLoader


class DHS_Wiki_Dataset(Dataset):
    """Data set class for the IMR-Maternal Education Level DHS-Wikipedia project"""

    def __init__(self, DHS_csv_file,
                 emb_root_dir, cluster_rank_csv_path, emb_dim=300, n_articles=5, include_dists=True,
                 country_subset=None, task='IMR',
                 transforms=None):
        """
        Args:
            HS_csv_file (string): Path to the csv file with DHS cluster values for IMR,
                as normalized rates, and maternal education level, as percents.
            emb_root_dir (string): Directory with all Wikipedia article embeddings.
            transforms (callable, optional): List of optional transforms to be applied
                consecutively on a sample.
        """
        self.DHS_frame = pd.read_csv(DHS_csv_file)
        self.emb_root_dir = emb_root_dir
        self.emb_dim = emb_dim
        self.n_articles = n_articles
        self.cluster_rank_csv_path = cluster_rank_csv_path
        self.cluster_article_rank_dist = pd.read_csv(self.cluster_rank_csv_path)
        self.include_dists = include_dists
        self.country_subset = country_subset
        self.task = task

        self.transforms = transforms
        if self.country_subset:
            self.subset(self.country_subset)

    def __len__(self):
        return len(self.DHS_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.task == 'IMR':
            outcome = self.DHS_frame['imr'].iloc[idx]
            outcome = np.array([outcome]).astype('float32')
        elif self.task == 'MatEd':
            outcome = self.DHS_frame[['pct_no_education',
                                     'pct_primary_education',
                                     'pct_secondary_education',
                                     'pct_higher_education']].iloc[idx]
            outcome = np.array([outcome]).astype('float32').squeeze()
            outcome = outcome/np.sum(outcome)

        if self.include_dists:
            article_embeddings = np.zeros((self.n_articles, self.emb_dim+1))
        else:
            article_embeddings = np.zeros((self.n_articles, self.emb_dim))

        # convert index into data set to index into DHS clusters
        idx_cluster = self.DHS_frame.id.iloc[idx]
        for k in range(self.n_articles):
            # convert index into DHS clusters into index into article embeddings
            article_idx = self.cluster_article_rank_dist['id_knn_{}'.format(k)].iloc[idx_cluster]
            emb = np.load(os.path.join(self.emb_root_dir, str(article_idx) + '.npy'))

            if self.include_dists:
                d = self.cluster_article_rank_dist['dist_{}'.format(k)].iloc[idx_cluster]
                article_embeddings[k] = np.concatenate([emb, np.expand_dims(d, axis=0)])
            else:
                article_embeddings[k] = emb

        if self.include_dists:
            article_embeddings = article_embeddings.reshape((self.n_articles * (self.emb_dim+1),)).astype('float32')
        else:
            article_embeddings = article_embeddings.reshape((self.n_articles * self.emb_dim,)).astype('float32')

        sample = {'x': article_embeddings, 'y': outcome}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def subset(self, countries):
        # country: list of str matching one of the country names in the DHS data set, specifically in the .csv file at the path
        #          locations given for the train/val/test sets
        # country_idx = self.train_set.country.isin(countries)
        self.DHS_frame = self.DHS_frame.loc[self.DHS_frame.country.isin(countries)]

def make_hp_dir(hp_dict):
    return str(hp_dict).translate(''.maketrans({'\'': '', ' ': '_', '{': '', '}': '', ',': '', '[': '', ']': '', ':':''}))

class MyDataLoader:
    def __init__(self, modes, train_path, val_path, test_path, country_subset = None, args=None, preload=True, seed=0, K=10):
        """
        Loads batches of data from given data sets

        :param modes: dict of keys giving modes to load in multimodal data, includes both input and output data.
            Possible options:
                IMR:        infant mortality rate
                MEL:        maternal education level
                wiki_embs:  Wikipedia article embeddings
                GUF:        Global Urban Footprint images centered on DHS clusters

        :param train_path:
        :param val_path:
        :param test_path:
        :param preload:
        """

        self.modes = modes
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self.train_set_save = pd.read_csv(self.train_path)
        self.val_set_save = pd.read_csv(self.val_path)
        self.test_set_save = pd.read_csv(self.test_path)
        self.country_subset = country_subset
        if country_subset:
            self.subset(country_subset)
        else:
            self.train_set = self.train_set_save
            self.val_set = self.val_set_save
            self.test_set = self.test_set_save

        # preloading by default for now
        self.article_embeddings_dir = os.path.join(os.curdir, 'raw', 'wikipedia', 'doc2vec_embeddings')
        self.cluster_article_rank_dist_path = os.path.join(os.curdir, 'processed',
                                                           'ClusterNearestArticles_Rank_Dist.csv')
        self.cluster_article_rank_dist = pd.read_csv(self.cluster_article_rank_dist_path)
        self.preload = preload
        # preload article embeddings

        self.emb_dim = 300 # args.emb_dim
        self.K = K
        # if self.preload:
        #     # EVENTUALLY GET TOTAL NUMBER OF RELEVANT GEOLOCATED ARTICLES(I.E. ONES THAT ARE THE NEAREST NEIGHBORS TO
        #     # ANY CLUSTER) STORED FOR PREALLOCATION
        #     self.article_embeddings = np.zeros((*self.cluster_article_rank_dist.shape, self.emb_dim))
        #     for i, idx in enumerate():
        #         for k in range(n_articles):
        #             article_idx = self.cluster_article_rank_dist.loc[i, 'id_knn_'.format(k)]
        #             article_embeddings[i, k] = np.load(os.path.join(self.article_embeddings_dir, article_idx, '.npy'))
        # else:
        #     self.article_embeddings = None

        # self.train_indices_left = self.train_set['id'].to_numpy()  #np.arange(len(self.train_set))
        # self.val_indices_left = self.val_set['id'].to_numpy()  #np.arange(len(self.val_set))
        # self.test_indices_left = self.test_set['id'].to_numpy()  #np.arange(len(self.test_set))
        self.train_indices_left = np.arange(len(self.train_set))
        self.val_indices_left = np.arange(len(self.val_set))
        self.test_indices_left = np.arange(len(self.test_set))
        np.random.shuffle(self.train_indices_left)
        np.random.shuffle(self.val_indices_left)
        np.random.shuffle(self.test_indices_left)
        self.i_train = 0
        self.i_val = 0
        self.i_test = 0

    def sample_batch(self, batchsize, batch_type, device, dtype):
        # sample batches from given data set without replacement epoch by epoch
        assert (batch_type == 'train' or batch_type == 'val' or batch_type == 'test'), \
                                        'Batch type must "train", "val", or "test"'

        if batch_type == 'train':
            data_set = self.train_set
            indices_left = self.train_indices_left
            i = self.i_train
        elif batch_type == 'val':
            data_set = self.val_set
            indices_left = self.val_indices_left
            i = self.i_val
        elif batch_type == 'test':
            data_set = self.test_set
            indices_left = self.test_indices_left
            i = self.i_test

        if len(indices_left) - i >= batchsize:
            batch = data_set.iloc[indices_left[i:i+batchsize]]
            print(i)
            print(self.i_train)
            print(batchsize)
            print(len(indices_left))

            if 'embeddings' in self.modes:
                X = self.get_article_embeddings(cluster_idx=data_set.id.iloc[indices_left[i:i+batchsize]].to_numpy(),
                                                n_articles=self.K,
                                                emb_dim=300, include_dists=('distances' in self.modes))

            i += batchsize
            self.epoch_flag = False
        else:
            np.random.shuffle(indices_left)
            i = 0
            batch = data_set.iloc[indices_left[i:i+batchsize]]
            self.epoch_flag = True

            if 'embeddings' in self.modes:
                X = self.get_article_embeddings(cluster_idx=data_set.id.iloc[indices_left[i:i+batchsize]].to_numpy(),
                                                n_articles=self.K,
                                                emb_dim=300, include_dists=('distances' in self.modes))
            i += batchsize

        if batch_type == 'train':
            self.i_train = i
        elif batch_type == 'val':
            self.i_val = i
        elif batch_type == 'test':
            self.i_test = i

        X = torch.Tensor(X).to(device=device, dtype=dtype)  # move to device, e.g. GPU
        if 'IMR' in self.modes:
            y = batch['imr'].to_numpy()
            y = torch.Tensor(y).to(device=device, dtype=torch.float32)
        elif 'MatEd' in self.modes:
            y = batch[['pct_no_education', 'pct_primary_education',
                      'pct_secondary_education', 'pct_higher_education']].to_numpy()/100.0
            y = torch.Tensor(y).to(device=device, dtype=torch.long)

        return X, y

    def subset(self, countries):
        # country: list of str matching one of the country names in the DHS data set, specifically in the .csv file at the path
        #          locations given for the train/val/test sets
        # country_idx = self.train_set.country.isin(countries)
        self.train_set = self.train_set_save.loc[self.train_set_save.country.isin(countries)]
        self.val_set = self.val_set_save.loc[self.val_set_save.country.isin(countries)]
        self.test_set = self.test_set_save.loc[self.test_set_save.country.isin(countries)]

    def get_article_embeddings(self, cluster_idx, n_articles, emb_dim=300, include_dists=True):
        # return numpy array of shape (n_clusters, n_articles, emb_dim (+1)) with article embeddings for embeddings of
        # n_articles closest Wikipedia articles, optionally with scaleless cluster-article Haversine distances

        # assume article embeddings have same file name as index with .npy attached
        article_embeddings = np.zeros((len(cluster_idx), n_articles, emb_dim + (1 if include_dists else 0)))
        for i, idx in enumerate(cluster_idx):
            for k in range(n_articles):
                # TODO: make sure csv indices are consistent across files
                article_idx = self.cluster_article_rank_dist.loc[idx, 'id_knn_{}'.format(k)]
                emb = np.load(os.path.join(self.article_embeddings_dir, str(article_idx) + '.npy'))

                d = self.cluster_article_rank_dist.loc[idx, 'dist_{}'.format(k)]
                if include_dists:
                    article_embeddings[i, k] = np.concatenate([emb, np.expand_dims(
                                        self.cluster_article_rank_dist.loc[idx, 'dist_{}'.format(k)], axis=0)])
                else:
                    article_embeddings[i, k] = emb
        return article_embeddings


def make_train_val_test_split(data_path, split_path, split, country_balanced=True):
    '''
    # takes in DHS CSV file, splits it randomly along rows into train, val, and test sets, and then saves the resulting
    # data subsets into CSV files into the split_path directory

    :param data_path: path to unsplit data CSV file
    :param split_path: directory to path to save data split SCV files (given same name as data file with suffixes
                        '_train', '_val', '_test'
    :param split: list of 3 integer values summing to 1 giving the train/val/test split in that order
    :param country_balanced: whether to enforce split within each country
    :return: the data split into arrays
    '''

    data = pd.read_csv(data_path)
    if country_balanced:
        train_data = pd.DataFrame()
        val_data = pd.DataFrame()
        test_data = pd.DataFrame()
        for country in data.country.unique():
            data_country = data.loc[data.country == country]
            n_data_country = len(data_country)
            n_train = int(split[0] * n_data_country)
            n_val = int(split[1] * n_data_country)
            print('{}: Train: {}  Val: {}  Test: {}'.format(country, n_train, n_val, n_data_country-n_train-n_val))
            # test set is remaining data

            idx = np.random.permutation(n_data_country)
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]
            train_data = train_data.append(data_country.iloc[train_idx], ignore_index=True)
            val_data = val_data.append(data_country.iloc[val_idx], ignore_index=True)
            test_data = test_data.append(data_country.iloc[test_idx], ignore_index=True)

    else:
        n_data = len(data)
        n_train = int(split[0]*n_data)
        n_val = int(split[1]*n_data)
        # test set is remaining data

        idx = np.random.permutation(n_data)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]

        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        test_data = data.iloc[test_idx]

    _, name = os.path.split(data_path)

    train_path = os.path.join(split_path, name)[:-4] + '_train.csv'
    val_path = os.path.join(split_path, name)[:-4] + '_val.csv'
    test_path = os.path.join(split_path, name)[:-4] + '_test.csv'

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))

    assert len(data) == len(train_data.append(val_data).append(test_data)) and \
            len(train_data.append(val_data).append(test_data)) == len(train_data['cluster_id'].unique()) + \
                                                                      len(val_data['cluster_id'].unique()) + \
                                                                          len(test_data['cluster_id'].unique()), \
                                                                    'Split not formed correctly; overlap between ' \
                                                                    'train/val/test sets'

    train_data.to_csv(train_path)
    val_data.to_csv(val_path)
    test_data.to_csv(test_path)

    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)

    print('Number of train points: {}\nValidation points: {}\nTest points: {}'.format(
                                len(train_data), len(val_data), len(test_data)))

    return train_data, val_data, test_data

def make_country_cross_comparison(countries):

    # countries: list of str country names
    country_set = []
    for country in countries:
        country_set.append(country)
    country_set.append(countries)
    countries_train, countries_test = country_set, country_set
    return countries_train, countries_test

def validate_article_embeddings(path_embeddings, path_article_list, path_validated_articles_save):
    article_df = pd.read_csv(path_article_list)
    embedding_file_names = os.listdir(path_embeddings)
    num_valid = 0
    num_total_listed = len(article_df)
    validated_article_idx = np.zeros(num_total_listed).astype(np.bool)
    # COULD INSTEAD CONVERT INDICES ON BOTH SIDES TO INTEGER LISTS, TAKEN SET INTERSECTION, THEN INDEXED INTO ARRAY
    for idx in range(len(embedding_file_names)):
        id_file = int(embedding_file_names[idx][:-4])
        mask = article_df.id==id_file
        if mask.any(axis=None):
            validated_article_idx[mask.to_numpy().argmax()] = True
            num_valid += 1

    validated_articles = article_df.loc[validated_article_idx]
    validated_articles.to_csv(path_validated_articles_save)
    print('{} valid article embeddings. {} total articles listed in given CSV.'.format(num_valid, num_total_listed))
    return validated_articles


def compute_nearest_coordinates(queries, neighbor_coords, path='', K=25, save=True):
    """
    computes the <K> nearest <neighbors> for each coordinate in <queries>.
    Also saves the resulting dataframe as a .csv at the given path
    :param queries: Pandas dataframe with lat and lon columns (in degrees) giving coordinates to get nearest neighbors of
    :param neighbor_coords: Pandas dataframe with lat and lon columns giving coordinates of set of potential neighbor
                        coordinates
    :param path: str, full path to save .csv of resulting dataframe of nearest neighbors
    :param K: number of nearest neighbors to determine for each query

    :return kNN: Pandas dataframe with ID column giving ID of query with columns '1', to 'K' giving the IDs of the
                neighbors, columns '1_dist' to 'K_dist' giving the respective Haversine distances to those neighbors
    """
    assert (os.path.exists(os.path.split(path)[0]) and os.path.split(path)[1]) or \
            not save, 'Must give proper file path to save distance rankings'

    neighbor_coords_np = neighbor_coords[['lat', 'lon']].to_numpy()*np.pi/180.0
    queries_np = queries[['lat', 'lon']].to_numpy()*np.pi/180.0

    knn = neighbors.NearestNeighbors(n_neighbors=K, metric='haversine')
    knn.fit(neighbor_coords_np)
    dist, idx = knn.kneighbors(queries_np)

    cols = ['id'] + ['id_knn_{}'.format(i) for i in range(K)] + ['dist_{}'.format(i) for i in range(K)]
    neighbor_idx = neighbor_coords['id'].to_numpy()[idx]
    data = np.concatenate([np.expand_dims(queries['id'].to_numpy(), axis=1), neighbor_idx, dist], axis=1)
    type_dict = {'id_knn_{}'.format(i): 'int32' for i in range(K)}
    type_dict['id'] = 'int32'
    nn_rank_dist = pd.DataFrame(data=data, columns=cols).astype(type_dict)

    if save:
        nn_rank_dist.to_csv(path)

    return nn_rank_dist

def main():
    # MAKE SURE TO ADD 'id' column of integer indices indexing over clutsers in ClusterLevelCombined_5yrIMR_MatEd.csv
    # and to ClusterCoordinates.csv (and eventually to every CSV with processed data) to track coordinates uniquely
    # across all transformations

    # run options:
    # run data-processing functions:
    # run = 'validate_articles'  # output CSV with articles IDs and article coordinates for each article that has a
                                # corresponding .npy file
    run = 'compute_nearest_articles' # compute array of indices and distances of K nearest articles to all DHS clusters
    # run = 'torch_batch' # form train/val/test split of DHS clusters

    # tests:
    # run = 'batch'  # test full batching
    # run = 'country_subset'  # test ability to subset data by sets of countries
    # run = 'fetch_embeddings' # test article embedding load process
    # run = 'neigh' # run tests of nearest neighbors
    # run = 'batch_sample' # run tests of batching
    # run = 'all' # run all tests. Does not run non-testing code, such as that for forming a train/val/test split

    if run == 'form_split':
        data_path = os.path.join(os.path.curdir, 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv')
        split_path = os.path.join(os.path.split(data_path)[0], 'split')
        train_val_test_split = [0.7, 0.15, 0.15]
        make_train_val_test_split(data_path, split_path, train_val_test_split)

    if run == 'validate_articles':
        path_embeddings = os.path.join(os.curdir, 'raw', 'wikipedia', 'doc2vec_embeddings')
        path_article_list = os.path.join(os.curdir, 'raw', 'wikipedia', 'All_Article_Coordinates.csv')
        if os.path.split(path_embeddings)[1] == 'doc2vec_embeddings':
            path_validated_articles_save = os.path.join(os.curdir, 'raw', 'wikipedia', 'All_Article_Coordinates_validated.csv')
        else:
            path_validated_articles_save = os.path.join(os.curdir, 'raw', 'wikipedia', 'All_Article_Coordinates_validated_toy.csv')

        articles = validate_article_embeddings(path_embeddings, path_article_list, path_validated_articles_save)
        print(articles)

    if run == 'compute_nearest_articles':
        K = 25
        queries = pd.read_csv(os.path.join(os.curdir, 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv'))
        neighs = pd.read_csv(os.path.join(os.curdir, 'raw', 'wikipedia', 'All_Article_Coordinates_validated.csv'))

        print('Sample queries:\n{}'.format(queries[['lat', 'lon']].head()))
        print('Sample neighbor coordinates:\n{}'.format(neighs.head()))
        save = True
        save_path = os.path.join(os.curdir, 'processed', 'ClusterNearestArticles_Rank_Dist.csv')
        rank_dist = compute_nearest_coordinates(queries, neighs, K=K, save=save, path=save_path)
        if save:
            print('Saved CSV of indices and distances of {} nearest neighbors to queries to {}'.format(K, save_path))

    # test batching with torch DataLoader class
    if run == 'torch_batch':
        article_embeddings_dir = os.path.join(os.curdir, 'raw', 'wikipedia', 'doc2vec_embeddings')
        cluster_article_rank_dist_path = os.path.join(os.curdir, 'processed',
                                                      'ClusterNearestArticles_Rank_Dist.csv')

        hps = []
        n_articles_nums = [1, 3]
        tasks = ['IMR', 'MatEd']
        countries_train_set = [None, ['Rwanda']]
        with_dists_hp = [False, True]
        emb_dims = [300]
        batch_sizes = {phase: size for phase, size in zip(['train', 'val', 'test'], [256, 1000, 1000])}

        for n_articles in n_articles_nums:
            for task in tasks:
                for countries_train in countries_train_set:
                    for emb_dim in emb_dims:
                        for include_dists in [False, True]:
                            hps.append(({'n_articles': n_articles, 'task': task,
                                         'countries_train': countries_train,
                                         'emb_dim': emb_dim, 'include_dists': include_dists}))

        idx_to_check = [0, 5, 10, 16, 21]

        for hp in hps:
            print('running with ')
            print(hp)

            task = hp['task']
            n_articles = hp['n_articles']
            modes = [task] + ['embeddings'] + ['distances']
            country_train = hp['countries_train']

            task = hp['task']
            n_articles = hp['n_articles']
            modes = [task] + ['embeddings'] + ['distances']
            country_train = hp['countries_train']
            include_dists = hp['include_dists']

            train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
            # val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
            # test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

            datasets = {phase: DHS_Wiki_Dataset(DHS_csv_file=train_path,  # test only with train set
                                                  emb_root_dir=article_embeddings_dir,
                                                  cluster_rank_csv_path=cluster_article_rank_dist_path,
                                                  emb_dim=hp['emb_dim'], n_articles=n_articles, include_dists=hp['include_dists'],
                                                  country_subset=country_train, task=task,
                                                  transforms=None)
                        for phase in ['train', 'val', 'test']}
            data_loaders = {phase: DataLoader(datasets[phase], batch_size=batch_sizes[phase], shuffle=True, num_workers=0)
                            for phase in ['train', 'val', 'test']}

            DHS_train_frame = pd.read_csv(train_path)
            if country_train:
                DHS_train_frame = DHS_train_frame.loc[DHS_train_frame.country.isin(countries_train)]

            cluster_article_rank_dist_frame = pd.read_csv(cluster_article_rank_dist_path)

            for idx in idx_to_check:
                item = datasets['train'].__getitem__(idx)
                x = item['x']
                y = item['y']

                # check batch shapes
                # assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                # assert x.shape == torch.Size((hp['emb_dim']+1,)) if hp['include_dists'] \
                #             else x.shape == torch.Size((hp['emb_dim'],))
                # assert y.shape == torch.Size((1,))
                assert x.shape == (n_articles*(hp['emb_dim']+1),) if hp['include_dists'] \
                            else x.shape == (n_articles*hp['emb_dim'],)
                if task == 'IMR':
                    assert y.shape == (1,)
                elif task == 'MatEd':
                    assert y.shape == (4,)

                #, Unnamed: 0, cluster_id, country, svy_yr_ed, nmothers, lat, lon, pct_no_education, pct_primary_education, pct_secondary_education, pct_higher_education, imr, yrgroup_imr, id
                # (0, 14, 'AM2015_111', 'Armenia', 2015, 11, 40.049908123899996, 44.2166119221, 0.0, 0.272727272727273, 0.545454545454545, 0.18181818181818202, 0.0, '2011 - 2015', 14)

                DHS_item_true = DHS_train_frame.iloc[idx]

                # check batch item values for selected indices
                if hp['task'] == 'IMR':
                    y_true = np.array([DHS_item_true['imr']]).astype('float32')
                    assert y.item() == y_true
                elif hp['task'] == 'MatEd':
                    y_true = np.array([DHS_item_true['pct_no_education',
                                             'pct_primary_education',
                                             'pct_secondary_education',
                                             'pct_higher_education']]).astype('float32')
                    y_true = y_true/np.sum(y_true)
                    assert np.all(y.item() == y_true)

                idx_cluster = DHS_train_frame.id.iloc[idx]
                if include_dists:
                    embs = np.zeros((n_articles, emb_dim+1))
                else:
                    embs = np.zeros((n_articles, emb_dim))
                for n in range(n_articles):
                    emb_id = int(cluster_article_rank_dist_frame.iloc[idx_cluster]['id_knn_{}'.format(n)])
                    emb = np.load(os.path.join(article_embeddings_dir, str(emb_id) + '.npy'))
                    if include_dists:
                        d = cluster_article_rank_dist_frame.iloc[idx_cluster]['dist_{}'.format(n)]
                        embs[n] = np.concatenate([emb, np.array([d])])
                    else:
                        embs[n] = emb

                x_true = embs.reshape(-1).astype('float32')
                assert np.all(x == x_true)
        print('All tests passed.')

    if run == 'batch' or run == 'all':
        train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
        val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
        test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

        modes = ['IMR', 'embeddings', 'distances']
        country_subset = None
        batchsize = 10
        emb_dim = 300
        n_articles = 10
        data_loader = MyDataLoader(modes, train_path, val_path, test_path, K=n_articles,
                                 country_subset=country_subset)
        X, y = data_loader.sample_batch(batchsize=batchsize, batch_type='val')
        assert X.shape == (batchsize, n_articles, (emb_dim + 1))
        if 'IMR' in modes:
            assert y.shape == (batchsize,)
        elif 'MatEd' in modes:
            assert y.shape == (batchsize, 4)

    if run == 'country_subset' or run == 'all':
        train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
        val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
        test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

        modes = []
        country_subset = ['Rwanda', 'Angola']
        data_loader = MyDataLoader(modes, train_path, val_path, test_path, country_subset=country_subset)
        assert data_loader.train_set.country.isin(country_subset).all() and \
               data_loader.val_set.country.isin(country_subset).all() and \
               data_loader.test_set.country.isin(country_subset).all(), 'Country subsetting allowing other items beyond given countries'
        print(data_loader.train_set.head())
        print(data_loader.val_set.head())
        print(data_loader.test_set.head())

    if run == 'fetch_embeddings' or run == 'all':
        train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
        val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
        test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

        modes = []
        data_loader = MyDataLoader(modes, train_path, val_path, test_path)

        start = time.time()

        range_clusters = 5000
        n_clusters = 500
        n_articles = 25
        emb_dim = 300
        cluster_ids = np.random.choice(range_clusters, n_clusters, replace=False)
        article_embeddings = data_loader.get_article_embeddings(cluster_ids, n_articles, emb_dim=emb_dim)
        assert article_embeddings.shape == (n_clusters, n_articles, emb_dim), \
                'article_embeddings array with shape {} does not match shape {}'.format(article_embeddings.shape,
                                                                            (n_clusters, n_articles, emb_dim))
        end = time.time()
        print(article_embeddings[0][0])
        print('Time to load {} embeddings: {} s\ngiving {} embeddings loaded/second.'.format(n_clusters*n_articles,
                                                                    end-start, n_clusters*n_articles/(end-start)))

    if run == 'neigh' or run == 'all':
        # test of computation of nearest articles
        neighs  = pd.DataFrame(data= np.array([[0, 60, 50],
                                               [3, 60, 60],
                                               [5, 60, 60],
                                               [7, 25, 60],
                                               [9, 60, -25]]), columns=['id', 'lat', 'lon'])

        print(neighs)

        queries = pd.DataFrame(data= np.array([[0, 60, 45],
                                               [1, 60, 60],
                                               [3, 30,  60],
                                               [8, 70, -30]]), columns=['id', 'lat', 'lon'])
        K = 3

        rank_dist = compute_nearest_coordinates(queries, neighs, K=K, save=False)
        print('Ranks and distances to queries: \n{}'.format(rank_dist))

    # tests of batch sampling
    if run == 'batch_sample' or run == 'all':
        train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
        val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
        test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

        modes = []
        data_loader = MyDataLoader(modes, train_path, val_path, test_path)
        batchsize = 5
        batch_type = 'train'
        n_epoch_points_init = len(data_loader.train_indices_left)
        # print("Training data points left initially: {}".format(n_epoch_points_init))
        batch = data_loader.sample_batch(batchsize, batch_type)
        n_epoch_points_after_batch = len(data_loader.train_indices_left)
        # print("Training data points left after batch: {}".format(n_epoch_points_after_batch))
        assert n_epoch_points_init == n_epoch_points_after_batch + batchsize, 'batch points left not decreasing correctly after batch taken'
        print('Working: Sampling batch decreases data points remaining')
        print(batch)

        for i in range(int(n_epoch_points_init//batchsize*1.5)):
            if len(data_loader.train_indices_left) < batchsize:
                n_epoch_points_init = len(data_loader.train_indices_left)
                batch = data_loader.sample_batch(batchsize, batch_type)
                n_epoch_points_after_batch = len(data_loader.train_indices_left)
                assert n_epoch_points_after_batch == n_epoch_points_init + len(data_loader.train_set) - batchsize, \
                                    'batch resetting across epochs not working correctly'
                print('Working: Batch resetting across epochs')
            else:
                batch = data_loader.sample_batch(batchsize, batch_type)

if __name__ == '__main__':
    main()
