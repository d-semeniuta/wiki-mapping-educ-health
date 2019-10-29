import os
import numpy as np
import pandas as pd
from sklearn import neighbors
import time

class DataLoader:
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
        self.train_indices_left = np.random.shuffle(np.arange(len(self.train_set)))
        self.val_indices_left = np.random.shuffle(np.arange(len(self.val_set)))
        self.test_indices_left = np.random.shuffle(np.arange(len(self.test_set)))

    def sample_batch(self, batchsize, batch_type):
        # sample batches from given data set without replacement epoch by epoch
        assert (batch_type == 'train' or batch_type == 'val' or batch_type == 'test'), \
                                        'Batch type must "train", "val", or "test"'

        if batch_type == 'train':
            data_set = self.train_set
            indices_left = self.train_indices_left
        elif batch_type == 'val':
            data_set = self.val_set
            indices_left = self.val_indices_left
        elif batch_type == 'test':
            data_set = self.test_set
            indices_left = self.test_indices_left

        if len(indices_left) >= batchsize:
            # idx = np.random.choice(len(indices_left), batchsize, replace=False)
            batch.append(data_set.iloc[indices_left[self.i:self.i+batchsize]])
            self.i += batchsize
            # batch = data_set.iloc[indices_left[[idx]]]
            # indices_left = np.delete(indices_left, idx)
            self.epoch_flag = False
        else:
            # batch = data_set.iloc[indices_left]
            # indices_left = data_set['id'].to_numpy()
            indices_left = np.random.shuffle(indices_left)
            self.i = 0

            # indices_left = np.arange(len(data_set))
            # BATCHES NOT EXCLUSIVE HERE
            # idx = np.random.choice(len(indices_left), batchsize - len(batch), replace=False)
            # idx = np.random.choice(len(indices_left), batchsize, replace=False)
            batch.append(data_set.iloc[indices_left[self.i:self.i+batchsize]])
            self.i += batchsize
            # indices_left = np.delete(indices_left, idx)
            self.epoch_flag = True

        if 'IMR' in self.modes:
            y = batch['imr'].to_numpy()
        elif 'MatEd' in self.modes:
            y = batch['pct_no_education', 'pct_primary_education',
                      'pct_secondary_education', 'pct_higher_education'].to_numpy()/100.0

        if 'embeddings' in self.modes:
            X = self.get_article_embeddings(cluster_idx=data_set.id[indices_left[idx]].to_numpy(), n_articles=self.K,
                                            emb_dim=300, include_dists=('distances' in self.modes))
            # X = X.reshape(batchsize, -1)

        # MAKE BATCH ONLY RETURN I/O PAIRS AS DICT OF NUMPY ARRAYS EVENTUALLY FOR MULTI-MODAL DATA?
        # HOPEFULLY WILL STILL BE ABLE TO JUST CONCATENATE NUMPY ARRAYS TOGETHER
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
    # run = 'compute_nearest_articles' # compute array of indices and distances of K nearest articles to all DHS clusters
    run = 'form_split' # form train/val/test split of DHS clusters

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

    if run == 'batch' or run == 'all':
        train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
        val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
        test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

        modes = ['IMR', 'embeddings', 'distances']
        country_subset = None
        batchsize = 10
        emb_dim = 300
        n_articles = 10
        data_loader = DataLoader(modes, train_path, val_path, test_path, K=n_articles,
                                 country_subset=country_subset)
        X, y = data_loader.sample_batch(batchsize=batchsize, batch_type='val')
        assert X.shape == (batchsize, n_articles, (emb_dim + 1))
        if 'IMR' in modes:
            assert y.shape == (batchsize,)
        elif 'MatEd' in modes:
            assert y.shape == (batchsize, 4)

    if run == 'country_subset' or 'all':
        train_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_train.csv')
        val_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_val.csv')
        test_path = os.path.join(os.path.curdir, 'processed', 'split', 'ClusterLevelCombined_5yrIMR_MatEd_test.csv')

        modes = []
        country_subset = ['Rwanda', 'Angola']
        data_loader = DataLoader(modes, train_path, val_path, test_path, country_subset=country_subset)
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
        data_loader = DataLoader(modes, train_path, val_path, test_path)

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
        data_loader = DataLoader(modes, train_path, val_path, test_path)
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

