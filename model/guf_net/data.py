import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pdb

from torchvision import transforms
# from torchvision.transforms import ToPILImage, CenterCrop
from skimage.io import imread


african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']

class GUFAfricaDataset(Dataset):

    def __init__(self, dhs_data_loc=None, cluster_image_dir=None, countries=african_countries):
        if isinstance(countries, str):
            countries = [countries]
        elif not isinstance(countries, list):
            Raise(TypeError('Must give either string or list'))
        if len(set(countries) - set(african_countries)) > 0:
            Raise(ValueError('Countries out of dataset'))

        proj_head = os.path.abspath('../../')
        if dhs_data_loc is None:
            dhs_data_loc = os.path.join(proj_head, 'data', 'processed',
                                        'ClusterLevelCombined_5yrIMR_MatEd.csv')
        combined_dhs = pd.read_csv(dhs_data_loc)

        self.combined_dhs = combined_dhs[combined_dhs['country'].isin(countries)]        

        if cluster_image_dir is None:
            self.cluster_image_dir = os.path.join(proj_head, 'data', 'raw',
                                                'nightlights', 'cluster_images')
        else:
            self.cluster_image_dir = cluster_image_dir

        # standardize image size This is ~ 5km x 5km angular
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
        # image = imread(img_nm)
        image = self.transform(imread(img_nm))


        ed_labels = ['no_education', 'primary_education', 'secondary_education',
                        'higher_education']
        ed_score = 0
        for i, label in enumerate(ed_labels):
            ed_score += i * cluster_row['pct_{}'.format(label)]

        imr = cluster_row['imr']
        return {'image': image, 'ed_score': ed_score, 'imr': imr}

def split_dataset(dataset, batch_size=16, validation_split=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset

def getDataLoaders(countries, batch_size, use_graph=False):
    datasets = {}
    for country in countries:
        country_set = GUFAfricaDataset(countries=country)
        datasets[country] = split_dataset(country_set, batch_size=batch_size)

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

    return data_loaders

if __name__ == '__main__':

    data_loaders = getDataLoaders(['Ghana', 'Zimbabwe', 'Kenya', 'Egypt'], 16)


    # guf = GUFAfricaDataset()
    # minx, miny = float('inf'), float('inf')
    # maxx, maxy = 0, 0
    # from tqdm import tqdm
    # for i in tqdm(range(10)):
    #     sample = guf[i]
    #     print(sample)
    #     x, y = sample['image'].shape
    #     minx = min(x, minx)
    #     miny = min(y, miny)
    #     maxx = max(x, maxx)
    #     maxy = max(y, maxy)
    #
    # print(minx, miny, maxx, maxy)
