import os

import pandas as pdb

from torch.utils.data import Dataset
from torch import from_numpy

african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']

class Doc2VecAfricaDataset(Dataset):

    def __init__(self, dhs_data_loc=None, doc2vec_feature_path = None, countries = african_countries):
        if isinstance(countries, str)
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

        if doc2vec_feature_path is None:
            doc2vec_feature_path = os.path.join(proj_head, 'data', 'processed', 'doc2vec_feature_set_two_hops.csv')

        self.doc2vec_embeddings = pd.read_csv(doc2vec_feature_path)

        def __len__(self):
            return len(self.combined_dhs)

        def __getitem__(self, idx):
            cluster_row = self.combined_dhs.iloc[idx]
            cluster_id = cluster_row['cluster_id']

            embedding = self.doc2vec_embeddings.loc[self.doc2vec_embeddings['id'] == int(cluster_id)].to_numpy()[1:]
            embedding = from_numpy(embedding)

            ed_labels = ['no_education', 'primary_education', 'secondary_education',
                            'higher_education']
            ed_score = 0
            for i, label in enumerate(ed_labels):
                ed_score += i * cluster_row['pct_{}'.format(label)]

            imr = cluster_row['imr']
            return {'embedding': embedding, 'ed_score': ed_score, 'imr': imr}
