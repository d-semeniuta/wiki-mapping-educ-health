import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch import from_numpy

from scipy import stats

african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']

class Graph2VecAfricaDataset(Dataset):

    def __init__(self, dhs_data_loc=None, graph2vec_feature_path = None, countries = african_countries):
        if isinstance(countries, str):
            countries = [countries]
        elif not isinstance(countries, list):
            Raise(TypeError('Must give either string or list'))
        if len(set(countries) - set(african_countries)) > 0:
            Raise(ValueError('Countries out of dataset'))

        proj_head = os.path.abspath('../../')
        if dhs_data_loc is None:
            dhs_data_loc = os.path.join(proj_head, 'data', 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv')

        combined_dhs = pd.read_csv(dhs_data_loc)
        self.combined_dhs = combined_dhs[combined_dhs['country'].isin(countries)]
        self.combined_dhs = self.combined_dhs[(np.abs(stats.zscore(self.combined_dhs.imr)) < 3)]

        if graph2vec_feature_path is None:
            graph2vec_feature_path = os.path.join(proj_head, 'data', 'processed', 'two_hop.csv')

        self.graph2vec_embeddings = pd.read_csv(graph2vec_feature_path, header=0)

    def __len__(self):
        return len(self.combined_dhs)

    def __getitem__(self, idx):
        cluster_row = self.combined_dhs.iloc[idx]
        cluster_id = cluster_row['id']

        embedding = list((self.graph2vec_embeddings.loc[self.graph2vec_embeddings.type == cluster_id]).to_numpy()[0])[1:]

        embedding = np.asarray(embedding)

        embedding = from_numpy(embedding).float()

        ed_labels = ['no_education', 'primary_education', 'secondary_education',
                        'higher_education']
        ed_score = 0
        for i, label in enumerate(ed_labels):
            ed_score += i * cluster_row['pct_{}'.format(label)]

        imr = cluster_row['imr']
        return {'embedding': embedding, 'ed_score': ed_score, 'imr': imr}

if __name__ == '__main__':
    doc = Graph2VecAfricaDataset(countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt'])
    from tqdm import tqdm
    for i in tqdm(range(10)):
        sample = doc[i]
        print(sample)
