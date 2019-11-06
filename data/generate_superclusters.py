import pandas as pd
import numpy as np
import os, sys
import argparse


repo_head_dir = os.path.abspath('../')
sys.path.append(repo_head_dir)

data_dir = os.path.join(repo_head_dir, 'data')
processed_dir = os.path.join(data_dir, 'processed')

from util.utils import compute_hav_dist_matrix, getCoordinateCentroid

def parseArgs():
    """ Get cluster size from the command line
    """
    parser = argparse.ArgumentParser(description='Generate Super Clusters.')
    parser.add_argument('--clustersize', type=int, default=10,
                       help='Size of clusters to generate')

    args = parser.parse_args()
    return args.clustersize

def nearestNeighborClustering(dists, cluster_sz=10, method=None):
    """Performs clustering using nearest neighbor

    Parameters
    ----------
    dists : np.array
        Distance matrix of all points
    cluster_sz : int
        Size of each cluster
    method : type
        Method of choosing initial cluster points. By default, max distance.
        Each cluster is formed around the point furthest away from all other points

    Returns
    -------
    list((np.array, (float, float)))
        List of arrays of indices of clusters and 'cost' of each cluster

    """
    if method is None:
        method = 'maxd'
    elif method not in ['maxd']:
        raise 'Invalid method'
    # orig_dists = np.copy(dists)
    num_points = len(dists)
    num_pts_leftover = num_points % cluster_sz

    clusters = []
    dist_costs = np.sum(dists, axis=1)

    while len(np.where(dist_costs > 0)[0]) >= cluster_sz:
        if num_pts_leftover > 0:
            # Even out the remainder over first few clusters
            this_cluster_sz = cluster_sz + 1
            num_pts_leftover -= 1
        else:
            this_cluster_sz = cluster_sz
        starting_pt = np.argmax(dist_costs)
        neighbor_dists = dists[starting_pt]
        closest_n = np.argpartition(neighbor_dists, cluster_sz)[:this_cluster_sz]
        dist_costs[closest_n] = -1
        intra_cluster_dists = dists[:, closest_n][closest_n, :]
        dists[:,closest_n] = 1e100

        mean_dist = np.mean(intra_cluster_dists)
        max_dist = np.max(intra_cluster_dists)
        km_per_rad = 6371
        clusters.append((closest_n, (mean_dist*km_per_deg, max_dist*km_per_deg)))

    return clusters

def generateSuperClusters(cluster_df, cluster_sz=10):
    countries = cluster_df.country.unique()
    country_dfs = []
    for country in countries:
        print('Now clustering for {}...'.format(country), flush=True)
        country_df = cluster_df.loc[cluster_df.country==country]
        country_code = country_df.cluster_id.iloc[0][:2]

        country_coords = country_df[['lat', 'lon']].to_numpy()
        dists = compute_hav_dist_matrix(country_coords, country_coords)
        # pre_dists = get_dist_matrix(country_df)
        # dists = np.copy(pre_dists)
        clusters = nearestNeighborClustering(dists, cluster_sz=cluster_sz)
        for i, (cluster, cost) in enumerate(clusters):
            sc_id = '{}_SC_{}'.format(country_code, i)
            cluster_coords = country_coords[cluster]
            centroid = getCoordinateCentroid(cluster_coords)
            cluster_idx = country_df.index[cluster]
            country_df.loc[cluster_idx, 'supercluster_id'] = sc_id
            country_df.loc[cluster_idx, 'sc_centroid_lat'] = centroid[0]
            country_df.loc[cluster_idx, 'sc_centroid_lon'] = centroid[1]
            country_df.loc[cluster_idx, 'mean_dist_supercluster_km'] = cost[0]
            country_df.loc[cluster_idx, 'max_dist_supercluster_km'] = cost[1]
        country_dfs.append(country_df)
    final_df = pd.concat(country_dfs)
    return final_df

def generateSuperClusterMetaData(supercluster_df):
    cols_to_use = ['country', 'supercluster_id', 'cluster_id', 'sc_centroid_lat',
                    'sc_centroid_lon', 'mean_dist_supercluster_km', 'max_dist_supercluster_km']
    agg_funcs = {'country': 'first', 'sc_centroid_lat': 'first', 'sc_centroid_lon': 'first', 'cluster_id': 'count',
                    'mean_dist_supercluster_km': 'first','max_dist_supercluster_km': 'first'}
    agg_df = supercluster_df[cols_to_use].groupby('supercluster_id', as_index=False).agg(agg_funcs)

    return agg_df.rename(columns={'cluster_id' : 'num_clusters'})

def runFull(cluster_sz, supercluster_csv_name):
    cluster_loc = os.path.join(processed_dir, 'ClusterCoordinates.csv')
    cluster_df = pd.read_csv(cluster_loc)
    cols_to_save = ['country', 'cluster_id', 'supercluster_id', 'lat', 'lon',
                    'sc_centroid_lat', 'sc_centroid_lon']
    supercluster_df = generateSuperClusters(cluster_df, cluster_sz=cluster_sz)
    supercluster_df[cols_to_save].to_csv(os.path.join(processed_dir, supercluster_csv_name), index=False)

    supercluster_md_df = generateSuperClusterMetaData(supercluster_df)
    supercluster_csv_name = 'SuperClusters_Size{}.csv'.format(cluster_sz)
    supercluster_md_df.to_csv(os.path.join(processed_dir, supercluster_csv_name), index=False)
    return supercluster_df[cols_to_save], supercluster_md_df

def aggregateClusterData(sc_assignments, sc_md, cluster_level_combined, cluster_sz):
    merged = pd.merge(sc_assignments, cluster_level_combined, on='cluster_id', suffixes=('', '_y'))
    merged.drop(list(merged.filter(regex='_y$')), axis=1, inplace=True)

    raw_cols = ['nbirth', 'ndeath', 'nmothers']
    for ed in ['no', 'primary', 'secondary', 'higher']:
        raw_cols.append('raw_{}_education'.format(ed))

    sc_data = merged.groupby('supercluster_id')[raw_cols].sum().reset_index()
    sc_data['imr'] = sc_data.ndeath / sc_data.nbirth

    for ed in ['no', 'primary', 'secondary', 'higher']:
        pct = 'pct_{}_education'.format(ed)
        raw = 'raw_{}_education'.format(ed)
        sc_data[pct] = sc_data[raw] / sc_data.nmothers

    sc_data = sc_md.merge(sc_data, on='supercluster_id', how='inner')

    sc_data.to_csv(os.path.join(processed_dir, 'SuperCluster_Size{}_AggData.csv'.format(cluster_sz)), index=False)

def main():

    cluster_sz = parseArgs()
    supercluster_csv_name = 'SuperClusterAssignments_Size{}.csv'.format(cluster_sz)
    sc_csv_loc = os.path.join(processed_dir, supercluster_csv_name)
    sc_metadata_csv_name = 'SuperClusters_Size{}.csv'.format(cluster_sz)
    sc_md_loc = os.path.join(processed_dir, sc_metadata_csv_name)

    if not os.path.isfile(sc_csv_loc):
        sc_assignments, sc_md = runFull(cluster_sz, supercluster_csv_name)
    else:
        print('Super Clusters Exist...')
        sc_assignments = pd.read_csv(sc_csv_loc)
        sc_md = pd.read_csv(sc_md_loc)

    cluster_lvl_loc = os.path.join(processed_dir, 'ClusterLevelCombined_5yrIMR_MatEd.csv')
    cluster_level_combined = pd.read_csv(cluster_lvl_loc)

    aggregateClusterData(sc_assignments, sc_md, cluster_level_combined, cluster_sz)

if __name__ == '__main__':
    main()
