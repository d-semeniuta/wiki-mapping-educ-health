import numpy as np
import pandas as pd

import os

data_dir = os.path.abspath('./raw/DHS')
processed_dir = os.path.abspath('./processed')
imr_5yr_loc = os.path.join(data_dir, 'InfantMortality_Cluster5Year.csv')
imr_1yr_loc = os.path.join(data_dir, 'InfantMortality_ClusterYear.csv')
mat_ed_loc = os.path.join(data_dir, 'MaternalEducation_cluster.csv')

def process_imr1yr(imr_1yr):
    imr_1yr['ndeath'] = imr_1yr.imr * imr_1yr.nbirth
    country_level_imr = imr_1yr.groupby(by=['country', 'child_birth_year'])[['ndeath', 'nbirth']].sum()
    country_level_imr['imr'] = country_level_imr.ndeath / country_level_imr.nbirth
    flat_country_level = country_level_imr.reset_index()
    flat_country_level.to_csv(os.path.join(processed_dir, 'CountryLevelIMR_1Year.csv'), index=False)

def process_imr5yr(imr_5yr):
    imr_5yr['ndeath'] = imr_5yr.imr * imr_5yr.nbirth
    country_level_imr = imr_5yr.groupby(by=['country', 'yrgroup'])[['ndeath', 'nbirth']].sum()
    country_level_imr['imr'] = country_level_imr.ndeath / country_level_imr.nbirth
    flat_country_level = country_level_imr.reset_index()
    flat_country_level.to_csv(os.path.join(processed_dir, 'CountryLevelIMR_5Year.csv'), index=False)
    return flat_country_level

def process_mat_ed(mat_ed):
    mat_ed['npercat'] = mat_ed.pct * mat_ed.ntot
    country_level_ed = mat_ed.groupby(by=['country', 'mother_ed_cat'])[['npercat']].sum()
    country_level_ed = country_level_ed.unstack()['npercat']
    country_level_ed['ntot'] = country_level_ed[0.0] + country_level_ed[1.0] + country_level_ed[2.0] + country_level_ed[3.0]
    country_level_ed['pct0'] = country_level_ed[0.0] / country_level_ed['ntot']
    country_level_ed['pct1'] = country_level_ed[1.0] / country_level_ed['ntot']
    country_level_ed['pct2'] = country_level_ed[2.0] / country_level_ed['ntot']
    country_level_ed['pct3'] = country_level_ed[3.0] / country_level_ed['ntot']
    country_level_ed = country_level_ed[['pct0', 'pct1', 'pct2', 'pct3', 'ntot']]

    country_level_ed_svy_yr = mat_ed.groupby(by=['country'])[['svy_yr']].agg(max)
    country_level_ed = pd.merge(country_level_ed, country_level_ed_svy_yr, on='country').reset_index()
    country_level_ed.to_csv(os.path.join(processed_dir, 'CountryLevelMaternalEducation_flat.csv'), index=False)
    return country_level_ed

def combine_imr_ed_country_level(country_level_imr, country_level_ed):
    country_level_imr_recent = country_level_imr.loc[country_level_imr.yrgroup=='2011-2015'].rename(columns={'yrgroup' : 'yrgroup_imr'})
    country_level_ed = country_level_ed.rename(columns={'ntot': 'nmothers', 'svy_yr' : 'svy_yr_ed'})
    country_level_combined = pd.merge(country_level_ed, country_level_imr_recent, on='country')
    country_level_combined['undereducated'] = country_level_combined['pct0'] + country_level_combined['pct1']
    country_level_combined['educated'] = country_level_combined['pct2'] + country_level_combined['pct3']
    country_level_combined.to_csv(os.path.join(processed_dir, 'CountryLevelCombined_5yr.csv'), index=False)

def combine_imr_ed_cluster_level(imr_5yr, mat_ed):

    cat_to_meaning = {
        'pct3': 'higher education', 'pct2': 'secondary education',
        'pct1': 'primary education', 'pct0': 'no education'
    }

    imr_5yr_recent = imr_5yr.loc[imr_5yr.yrgroup=='2011-2015'] \
        [['cluster_id', 'imr', 'yrgroup']].rename(columns={'yrgroup' : 'yrgroup_imr'})
    mat_ed_flat = mat_ed
    for cat in range(4):
        col = 'pct{}'.format(cat)
        meaning = cat_to_meaning[col].replace(' ', '_')
        mat_ed_flat.loc[mat_ed_flat.mother_ed_cat==cat, 'pct_{}'.format(meaning)] = mat_ed_flat.pct

    mat_ed_flat = mat_ed_flat.groupby('cluster_id').agg(max).fillna(0)
    mat_ed_flat = mat_ed_flat[['country', 'svy_yr', 'ntot', 'lat', 'lon',
                               'pct_no_education', 'pct_primary_education',
                               'pct_secondary_education', 'pct_higher_education']].reset_index() \
                               .rename(columns={'ntot': 'nmothers', 'svy_yr' : 'svy_yr_ed'})
    mat_ed_flat.to_csv(os.path.join(processed_dir, 'MaternalEducation_cluster_flat.csv'), index=False)

    cluster_level_combined = pd.merge(mat_ed_flat, imr_5yr_recent, on='cluster_id', how='inner')
    cluster_level_combined.to_csv(os.path.join(processed_dir, 'ClusterLevelCombined_5yrIMR_MatEd.csv'), index=False)

def build_cluster_info(imr_1yr, imr_5yr, mat_ed):
    df1 = mat_ed[['country', 'cluster_id', 'lat', 'lon']]
    df2 = imr_5yr[['country', 'cluster_id', 'lat', 'lon']]
    df3 = imr_1yr[['country', 'cluster_id', 'lat', 'lon']]
    clusters = pd.concat([df1,df2,df3]).drop_duplicates()
    clusters.to_csv(os.path.join(processed_dir, 'ClusterCoordinates.csv'), index=False)

def main():
    imr_1yr = pd.read_csv(imr_1yr_loc)
    imr_5yr = pd.read_csv(imr_5yr_loc)
    mat_ed = pd.read_csv(mat_ed_loc)
    
    process_imr1yr(imr_1yr)
    country_level_imr = process_imr5yr(imr_5yr)
    flat_country_level_ed = process_mat_ed(mat_ed)
    combine_imr_ed_country_level(country_level_imr, flat_country_level_ed)
    combine_imr_ed_cluster_level(imr_5yr, mat_ed)

    build_cluster_info(imr_1yr, imr_5yr, mat_ed)

if __name__ == '__main__':
    main()
