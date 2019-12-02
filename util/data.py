import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from skimage.io import imread

import torch

african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']

class CombinedAfricaDataset(Dataset):
    def __init__(self, dhs_data_loc=None, cluster_image_dir=None, vec_feature_path=None,
                    use_graph=False, countries=african_countries):
        if isinstance(countries, str):
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

        # guf data
        if cluster_image_dir is None:
            self.cluster_image_dir = os.path.join(proj_head, 'data', 'processed', 'guf')
        else:
            self.cluster_image_dir = cluster_image_dir

        self.use_graph = use_graph

        # wiki embeddings
        if self.use_graph:
            if graph2vec_feature_path is None:
                graph2vec_feature_path = os.path.join(proj_head, 'data', 'processed', 'two_hop.csv')

            self.graph2vec_embeddings = pd.read_csv(graph2vec_feature_path, header=0)
        else:
            doc2vec_path = os.path.join(proj_head, 'model', 'doc2vec', 'coord_articles_only_doc2vec.model')
            self.doc2vec = Doc2Vec.load(doc2vec_path)

            if nearest_articles_path is None:
                nearest_articles_path = os.path.join(proj_head, 'data', 'processed', 'nearest_articles.csv')

            self.nearest_articles = pd.read_csv(nearest_articles_path, sep=";", header = None)

    def __len__(self):
        return len(self.combined_dhs)

    def __getitem__(self, idx):
        cluster_row = self.combined_dhs.iloc[idx]
        cluster_id = cluster_row['cluster_id']
        img_nm = os.path.join(self.cluster_image_dir, '{}.png'.format(cluster_id))
        image = self.transform(imread(img_nm))

        if self.use_graph:
            cluster_index = cluster_row['id']
            embedding = list((self.graph2vec_embeddings.loc[self.graph2vec_embeddings.type == cluster_index]).to_numpy()[0])[1:]
            embedding = np.asarray(embedding)
            embedding = from_numpy(embedding).float()
        else:
            cluster_index = cluster_row['id']
            cols = self.nearest_articles.columns
            articles_row = self.nearest_articles.loc[self.nearest_articles[cols[0]] == cluster_index]
            titles = list(articles_row[cols[1:11]].to_numpy()[0])
            embedding = []
            for title in titles:
                temp_list = list(self.doc2vec.docvecs[title])
                embedding += temp_list
            dists = list(articles_row[cols[11:]].to_numpy()[0])
            embedding += dists
            embedding = np.asarray(embedding)
            embedding = from_numpy(embedding).float()

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

def getDataLoaders(countries, guf_path, vec_feature_path, batch_size, use_graph=False):
    def get_dataset(countries):
        return CombinedAfricaDataset(countries=countries, cluster_image_dir=guf_path,
                                        use_graph=use_graph, vec_feature_path=vec_feature_path)
    data_loaders = {}
    for country in countries:
        data = get_dataset(country)
        others = [c for c in countries if c is not country]
        others_data = get_dataset(others)
        data_loaders[country] = {}
        data_loaders[country]['train'], data_loaders[country]['val'] = split_dataset(data, batch_size=batch_size)
        data_loaders[country]['others'] = {}
        data_loaders[country]['others']['train'], data_loaders[country]['others']['val'] = split_dataset(others_data, batch_size=batch_size)
    alldata = get_dataset(countries)
    data_loaders['all'] = {}
    data_loaders['all']['train'], data_loaders['all']['val'] = split_dataset(alldata, batch_size=batch_size)

    return data_loaders
