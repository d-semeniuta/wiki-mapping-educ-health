import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, SubsetRandomSampler
from torchvision import transforms
from skimage.io import imread

from torch import from_numpy

from gensim.models.doc2vec import Doc2Vec

import torch

african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']

class CombinedAfricaDataset(Dataset):
    def __init__(self, dhs_data_loc=None, cluster_image_dir=None, vec_feature_path=None,
                    use_graph=False, countries=african_countries):
        if isinstance(countries, str):
            countries = [countries]
        elif not isinstance(countries, list):
            raise(TypeError('Must give either string or list'))
        if len(set(countries) - set(african_countries)) > 0:
            diff_countries = set(countries) - set(african_countries)
            raise(ValueError('Countries out of dataset: {}'.format(diff_countries)))

        proj_head = os.path.abspath('./')

        # ground truth
        if dhs_data_loc is None:
            dhs_data_loc = os.path.join(proj_head, 'data', 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv')
        combined_dhs = pd.read_csv(dhs_data_loc)
        self.combined_dhs = combined_dhs[combined_dhs['country'].isin(countries)]

        # guf data
        if cluster_image_dir is None:
            self.cluster_image_dir = os.path.join(proj_head, 'data', 'raw', 'nightlights', 'cluster_images')
        else:
            self.cluster_image_dir = cluster_image_dir

        # wiki embeddings
        graph2vec_feature_path = os.path.join(proj_head, 'data', 'processed', 'two_hop.csv')
        self.graph2vec_embeddings = pd.read_csv(graph2vec_feature_path, header=0)

        doc2vec_path = os.path.join(proj_head, 'model', 'doc2vec', 'coord_articles_only_doc2vec.model')
        self.doc2vec = Doc2Vec.load(doc2vec_path)
        nearest_articles_path = os.path.join(proj_head, 'data', 'processed', 'nearest_articles.csv')
        self.nearest_articles = pd.read_csv(nearest_articles_path, sep=";", header = None)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(64),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.combined_dhs)

    def __getitem__(self, idx):
        cluster_row = self.combined_dhs.iloc[idx]
        cluster_id = cluster_row['cluster_id']
        img_nm = os.path.join(self.cluster_image_dir, '{}.png'.format(cluster_id))
        image = self.transform(imread(img_nm))

        # graph embs
        cluster_index = cluster_row['id']
        graph_emb = list((self.graph2vec_embeddings.loc[self.graph2vec_embeddings.type == cluster_index]).to_numpy()[0])[1:]
        graph_emb = np.asarray(graph_emb)
        graph_emb = from_numpy(graph_emb).float()
        # doc embs
        cluster_index = cluster_row['id']
        cols = self.nearest_articles.columns
        articles_row = self.nearest_articles.loc[self.nearest_articles[cols[0]] == cluster_index]
        titles = list(articles_row[cols[1:11]].to_numpy()[0])
        doc_emb = []
        for title in titles:
            temp_list = list(self.doc2vec.docvecs[title])
            doc_emb += temp_list
        dists = list(articles_row[cols[11:]].to_numpy()[0])
        doc_emb += dists
        doc_emb = np.asarray(doc_emb)
        doc_emb = from_numpy(doc_emb).float()

        ed_labels = ['no_education', 'primary_education', 'secondary_education',
                        'higher_education']
        ed_score = 0
        for i, label in enumerate(ed_labels):
            ed_score += i * cluster_row['pct_{}'.format(label)]

        imr = cluster_row['imr']
        return {'doc_emb': doc_emb, 'graph_emb': graph_emb, 'image': image, 'ed_score': ed_score, 'imr': imr}


def split_dataset(dataset, batch_size=16, validation_split=0.2):
    np.random.seed(45)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset

def get_overfit_dataloaders(dataset, subsample_size=8):
    train_sampler = SubsetRandomSampler(list(range(subsample_size)))
    train_loader = DataLoader(dataset, batch_size=subsample_size,
                                        sampler=train_sampler)
    return train_loader

def getDataLoaders(countries, guf_path, vec_feature_path, batch_size, model_dir, use_graph=False, overfit=False):
    data_save_loc = os.path.join(model_dir, 'dataloaders.pth')
    if os.path.isfile(data_save_loc):
        print('Getting dataloaders from pickle...')
        return torch.load(data_save_loc)

    datasets = {}
    for country in countries:
        country_set = CombinedAfricaDataset(countries=country, cluster_image_dir=guf_path,
                                        use_graph=use_graph, vec_feature_path=vec_feature_path)
        datasets[country] = split_dataset(country_set, batch_size=batch_size)
    if overfit:
        country = countries[0]
        loader = get_overfit_dataloaders(datasets[country][0])
        data_loaders = {}
        data_loaders[country] = {
            'train' : loader,
            'val' : loader
        }
        return data_loaders
    data_loaders = {}
    for country in countries:
        train_data, val_data = datasets[country]
        data_loaders[country] = {}
        data_loaders[country]['train'] = DataLoader(train_data, batch_size=batch_size)
        data_loaders[country]['val'] = DataLoader(val_data, batch_size=batch_size)

        others = zip(*[datasets[c] for c in countries if c is not country])
        others = [ConcatDataset(d) for d in others]
        data_loaders[country]['others'] = {}
        data_loaders[country]['others']['train'] = DataLoader(others[0], batch_size=batch_size)
        data_loaders[country]['others']['val'] = DataLoader(others[1], batch_size=batch_size)

    alldata = zip(*[datasets[c] for c in countries])
    alldata = [ConcatDataset(d) for d in alldata]
    data_loaders['all'] = {}
    data_loaders['all']['train'] = DataLoader(alldata[0], batch_size=batch_size)
    data_loaders['all']['val'] = DataLoader(alldata[1], batch_size=batch_size)

    print('Saving dataloaders to pickle...')
    torch.save(data_loaders, data_save_loc)
    return data_loaders
