import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdal

import os
import subprocess


from util_gdal import GeoProps

data_dir = os.path.abspath('../data/raw')
imr_5yr_loc = os.path.join(data_dir, 'DHS', 'InfantMortality_Cluster5Year.csv')
imr_1yr_loc = os.path.join(data_dir, 'DHS', 'InfantMortality_ClusterYear.csv')
mat_ed_loc = os.path.join(data_dir, 'DHS', 'MaternalEducation_cluster.csv')

imr_5yr = pd.read_csv(imr_5yr_loc)

num_rows = len(imr_5yr)
num_rows_0imr = len(imr_5yr[imr_5yr.imr == 0])
print('Number of cluster measurements:', num_rows)
print('Number of cluster measurements with imr 0:', num_rows_0imr)
print('Percentage:', num_rows_0imr / num_rows)
print('Number of unique clusters:', imr_5yr['cluster_id'].nunique())
print('Number of countries surveyed:', imr_5yr['country'].nunique())

# open GDAL data set
filepath = r"../data/raw/GUF_Continent_Africa.tif"

# for reduced image data set (better for seeing all of Africa at once and getting a sense of the coordinate system)
# produces data set with each pixel corresponding to a tenth of a degree shift in lat/long on a side
# import os
# os.system('gdal_translate -r lanczos -tr 0.1 0.1  -co COMPRESS=LZW ../data/GUF_Continent_Africa.tif '
#           '../data/raw/GUF_Continent_Africa_tenth.tif')
reduced_filepath = r"../data/raw/GUF_Continent_Africa_tenth.tif"
gdal_cmd = 'gdal_translate -r lanczos -tr 0.1 0.1 -co COMPRESS=LZW'.split()
gdal_cmd.extend([filepath, reduced_filepath])
 # ../data/GUF_Continent_Africa.tif ../data/raw/GUF_Continent_Africa_tenth.tif'
subprocess.call(gdal_cmd)


# Open the file:
raster = gdal.Open(filepath)

if not raster:
    print("Raster file not loaded correctly.")

# import GDAL data into GeoProps object for ease of use
geo_prop = GeoProps()
geo_prop.import_geogdal(raster)

# IMR unique measurements: 125261, 85006 with imr 0, 30979 in Africa, 17784 with imr 0 in Africa
# unique clusters: 42092 total, 10507 in Africa
# 25 countries total, 15 in Africa

# assume clusters are identical across different surveys
# unique_DHS_clusters = imr_5yr.drop_duplicates(['lat','lon'])
unique_DHS_clusters = imr_5yr
# current nightlights data set only covers Africa, must limit nightlights queries to coordinates within Africa
unique_DHS_clusters_Africa = unique_DHS_clusters.query('(lat < 37.3408) '
                                                       '& (lon > -25.360996)'
                                                       '& (lat > -34.821334)'
                                                       '& (lon < 63.4954491)')
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
# print(unique_DHS_clusters_Africa)
print('Number of unique clusters:', unique_DHS_clusters_Africa['cluster_id'].nunique())
print('Number of countries surveyed:', unique_DHS_clusters_Africa['country'].nunique())

mean_intensities = []
std_intensities = []

unique_DHS_clusters_Africa = unique_DHS_clusters_Africa.head(n=2)
print(unique_DHS_clusters_Africa['cluster_id'].nunique())
# 10507 unique clusters in Africa

for idx, cluster in unique_DHS_clusters_Africa.iterrows():
    # print(cluster)
    raster_img = geo_prop.get_coord_centered_img(cluster['lat'], cluster['lon'], 5, 5, raster,
                                                 filepath=data_dir + '/nightlights/' + cluster['cluster_id'] + '.png')

    mean_intensities.append(float(np.mean(raster_img)))
    std_intensities.append(float(np.std(raster_img)))

# add mean and standard deviation of intensities
unique_DHS_clusters_Africa_intensity = unique_DHS_clusters_Africa
unique_DHS_clusters_Africa_intensity['mean_intensity'] = mean_intensities
unique_DHS_clusters_Africa_intensity['std_intensity'] = std_intensities

unique_DHS_clusters_Africa_intensity.to_csv(data_dir + r'/InfantMortality_Cluster5year_intensities.csv')

test = pd.read_csv(data_dir + r'/InfantMortality_Cluster5year_intensities.csv')
# print(test)
