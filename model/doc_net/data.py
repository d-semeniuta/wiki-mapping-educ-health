import os

import pandas as pdb

from torch.utils.data import Dataset
from torch import from_numpy

from gensim.models.doc2vec import Doc2Vec

african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']

class Doc2VecAfricaDataset(Dataset):

    def __init__(self, dhs_data_loc=None, nearest_articles_path = None, countries = african_countries):
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

        doc2vec_path = os.path.join(proj_head, 'model', 'doc2vec', 'coord_articles_only_doc2vec.model')
        self.doc2vec = Doc2Vec.load(doc2vec_path)

        if nearest_articles_path is None:
            nearest_articles_path = os.path.join(proj_head, 'data', 'processed', 'nearest_articles.csv')

        self.nearest_articles = pd.read_csv(nearest_articles_path, sep=";")

        def __len__(self):
            return len(self.combined_dhs)

        def __getitem__(self, idx):
            cluster_row = self.combined_dhs.iloc[idx]

            cols = self.nearest_articles.columns
            articles_row = self.nearest_articles.loc[self.nearest_articles[cols[0]] == idx]
            titles = list(articles_row[cols[1:11]].to_numpy())
            embedding = []
            for title in titles:
                embedding += list(model.docvecs[title])
            dists = map(float,list(articles_row[cols[11:]].to_numpy()))
            embedding += dists

            ed_labels = ['no_education', 'primary_education', 'secondary_education',
                            'higher_education']
            ed_score = 0
            for i, label in enumerate(ed_labels):
                ed_score += i * cluster_row['pct_{}'.format(label)]

            imr = cluster_row['imr']
            return {'embedding': embedding, 'ed_score': ed_score, 'imr': imr}
