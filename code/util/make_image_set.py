import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdal
from tqdm import tqdm

import os
import subprocess


from util_gdal import GeoProps

DHS_data_dir = os.path.abspath('../data/raw/DHS')
data_dir = os.path.abspath('../data')

combined_dhs_loc = os.path.join(data_dir, 'processed', 'ClusterLevelCombined_5yrIMR_MatEd.csv')

combined_dhs = pd.read_csv(combined_dhs_loc)

# open GDAL data set
guf_dir = os.path.abspath('../data/raw/guf')
guf_out_path = os.path.abspath('../data/processed/guf/')
os.makedirs(guf_out_path, exist_ok=True)

GUF_filepath = os.path.join(guf_dir, "GUF_Continent_Africa.tif")

# Open the file:
raster = gdal.Open(GUF_filepath)

if not raster:
    print("Raster file not loaded correctly.")

# import GDAL data into GeoProps object for ease of use
geo_prop = GeoProps()
geo_prop.import_geogdal(raster)

african_countries = ['Angola', 'Benin', 'Burundi', 'Egypt', 'Ethiopia', 'Ghana', 'Kenya',
    'Lesotho', 'Malawi', 'Mozambique', 'Rwanda', 'Chad', 'Tanzania', 'Uganda', 'Zimbabwe']
combined_dhs_africa = combined_dhs[combined_dhs['country'].isin(african_countries)]
unique_DHS_clusters_Africa = combined_dhs_africa

print()
print('Africa:')
num_rows = len(unique_DHS_clusters_Africa)
num_rows_0imr = len(unique_DHS_clusters_Africa[unique_DHS_clusters_Africa.imr == 0])
print('Number of cluster measurements:', num_rows)
print('Number of cluster measurements with imr 0:', num_rows_0imr)
print('Percentage:', num_rows_0imr / num_rows)
print('Number of unique clusters:', unique_DHS_clusters_Africa['cluster_id'].nunique())
print('Number of countries surveyed:', unique_DHS_clusters_Africa['country'].nunique())

print("Unique cluster measurements in Africa:")
print(len(unique_DHS_clusters_Africa))
print('Number of unique clusters:', unique_DHS_clusters_Africa['cluster_id'].nunique())
print('Number of countries surveyed:', unique_DHS_clusters_Africa['country'].nunique())

mean_intensities = []
std_intensities = []

cluster = unique_DHS_clusters_Africa.iloc[0]
for idx, cluster in tqdm(unique_DHS_clusters_Africa.iterrows(), desc='Building Rasters',
                            total=len(unique_DHS_clusters_Africa)):
    raster_img = geo_prop.get_coord_centered_img(cluster['lat'], cluster['lon'], 6, 6, raster,
                                                 filepath=os.path.join(guf_out_path, cluster['cluster_id'] + '.png'))

    mean_intensities.append(float(np.mean(raster_img)))
    std_intensities.append(float(np.std(raster_img)))

# add mean and standard deviation of intensities
unique_DHS_clusters_Africa_intensity = unique_DHS_clusters_Africa
unique_DHS_clusters_Africa_intensity['mean_intensity'] = mean_intensities
unique_DHS_clusters_Africa_intensity['std_intensity'] = std_intensities

processed_dir = os.path.join(data_dir, 'processed')
unique_DHS_clusters_Africa_intensity.to_csv(os.path.join(data_dir, 'processed', 'combined_dhs_africa_guf_intensities.csv'),
                                            index=False)
