import os

import pandas as pd

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from skimage.io import imread

import torch

african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']

class CombinedAfricaDataset(Dataset):
    def __init__(self, dhs_data_loc=None, cluster_image_dir=None,
                    graph2vec_feature_path=None, doc2vec_feature_path=None,
                    use_graph=False, countries=african_countries):
        if isinstance(countries, str)
            countries = [countries]
        elif not isinstance(countries, list):
            Raise(TypeError('Must give either string or list'))
        if len(set(countries) - set(african_countries)) > 0:
            Raise(ValueError('Countries out of dataset'))

        proj_head = os.path.abspath('../')

        # ground truth
        if dhs_data_loc is None:
            dhs_data_loc = os.path.join(proj_head, 'data', 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv')
        combined_dhs = pd.read_csv(dhs_data_loc)
        self.combined_dhs = combined_dhs[combined_dhs['country'].isin(countries)]

        if use_graph:
            if graph2vec_feature_path is None:
                graph2vec_feature_path = os.path.join(proj_head, 'data', 'processed', 'graph2vec_feature_set_two_hops.csv')
            self.embeddings = pd.read_csv(graph2vec_feature_path)
        else:
            if doc2vec_feature_path is None:
                doc2vec_feature_path = os.path.join(proj_head, 'data', 'processed', 'doc2vec_feature_set_two_hops.csv')
            self.embeddings = pd.read_csv(doc2vec_feature_path)

    def __len__(self):
        return len(self.combined_dhs)

    def __getitem__(self, idx):
        cluster_row = self.combined_dhs.iloc[idx]
        cluster_id = cluster_row['cluster_id']
        img_nm = os.path.join(self.cluster_image_dir, '{}.png'.format(cluster_id))
        image = self.transform(imread(img_nm))

        embedding = self.embeddings.loc[self.embeddings['id'] == int(cluster_id)].to_numpy()
        embedding = torch.from_numpy(embedding)

        ed_labels = ['no_education', 'primary_education', 'secondary_education',
                        'higher_education']
        ed_score = 0
        for i, label in enumerate(ed_labels):
            ed_score += i * cluster_row['pct_{}'.format(label)]

        imr = cluster_row['imr']
        return {'emb': embedding, 'image': image, 'ed_score': ed_score, 'imr': imr}


def split_dataset(dataset, batch_size=16, validation_split=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=valid_sampler)
    return train_loader, val_loader

def getDataLoaders(params, countries, use_graph=False):
    data_loaders = {}
    for country in countries:
        data = CombinedAfricaDataset(countries=country, use_graph=use_graph)
        others = [c for c in countries if c is not country]
        others_data = CombinedAfricaDataset(countries=others, use_graph=use_graph)
        data_loaders[country] = {}
        data_loaders[country]['train'], data_loaders[country]['val'] = split_dataset(data, batch_size=params['batch_size'])
        data_loaders[country]['others'] = {}
        data_loaders[country]['others']['train'], data_loaders[country]['others']['val'] = split_dataset(others_data)
    alldata = CombinedAfricaDataset(countries=countries, use_graph=use_graph)
    data_loaders['all'] = {}
    data_loaders['all']['train'], data_loaders['all']['val'] = split_dataset(alldata)

    return data_loaders
