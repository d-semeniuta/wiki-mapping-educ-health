import os

import pandas as pd

from torch.utils.data import Dataset
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

if __name__ == '__main__':
    guf = GUFAfricaDataset()
    minx, miny = float('inf'), float('inf')
    maxx, maxy = 0, 0
    from tqdm import tqdm
    for i in tqdm(range(10)):
        sample = guf[i]
        print(sample)
    #     x, y = sample['image'].shape
    #     minx = min(x, minx)
    #     miny = min(y, miny)
    #     maxx = max(x, maxx)
    #     maxy = max(y, maxy)
    #
    # print(minx, miny, maxx, maxy)
