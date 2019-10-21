import pandas as pd
import numpy as np
import os, sys
import argparse


repo_head_dir = os.path.abspath('../')
sys.path.append(repo_head_dir)

from util.utils import haversine_dist

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
    list(np.array)
        List of arrays of indices of clusters

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
        assert(starting_pt in set(closest_n))
        dist_costs[closest_n] = -1
        intra_cluster_dists = dists[:, closest_n][closest_n, :]
        dists[:,closest_n] = 1e100

        mean_dist = np.mean(intra_cluster_dists)
        max_dist = np.max(intra_cluster_dists)
        km_per_deg = 6371
        clusters.append((closest_n, (mean_dist*km_per_deg, max_dist*km_per_deg)))

    return clusters

def get_dist_matrix(df):
    """Generates distance matrix between all points

    Parameters
    ----------
    df : Pandas.DataFrame
        dataframe of cluster coordinates, should contain lat and lon columns

    Returns
    -------
    np.array
        Symmetric matrix of haversine distances

    """
    all_coords = df[['lat', 'lon']].to_numpy()
    lat_vec, lon_vec = all_coords[:,0], all_coords[:,1]
    num_clusters = len(df)
    dists = np.zeros((num_clusters, num_clusters))
    for i, (this_lat, this_lon) in enumerate(zip(lat_vec, lon_vec)):
        dists[i] = haversine_dist(this_lat, this_lon, lat_vec, lon_vec)
    return dists

def generateSuperClusters(cluster_df, cluster_sz=10):
    countries = cluster_df.country.unique()
    country_dfs = []
    for country in countries:
        print('Now clustering for {}...'.format(country), flush=True)
        country_df = cluster_df.loc[cluster_df.country==country]
        country_code = country_df.cluster_id.iloc[0][:2]
        pre_dists = get_dist_matrix(country_df)
        dists = np.copy(pre_dists)
        clusters = nearestNeighborClustering(dists, cluster_sz=cluster_sz)
        for i, (cluster, cost) in enumerate(clusters):
            sc_id = '{}_SC_{}'.format(country_code, i)
            country_df.loc[country_df.index[cluster], 'supercluster_id'] = sc_id
            country_df.loc[country_df.index[cluster], 'mean_dist_supercluster_km'] = cost[0]
            country_df.loc[country_df.index[cluster], 'max_dist_supercluster_km'] = cost[1]
        country_dfs.append(country_df)
    final_df = pd.concat(country_dfs)
    return final_df

def generateSuperClusterMetaData(supercluster_df):
    cols_to_use = ['country', 'supercluster_id', 'mean_dist_supercluster_km',
                    'max_dist_supercluster_km', 'cluster_id']
    agg_funcs = {'country': 'max', 'mean_dist_supercluster_km': 'max',
                    'max_dist_supercluster_km': 'max', 'cluster_id' : 'count'}
    agg_df = supercluster_df[cols_to_use].groupby('supercluster_id', as_index=False).agg(agg_funcs)

    return agg_df.rename(columns={'cluster_id' : 'num_clusters'})

def main():
    data_dir = os.path.join(repo_head_dir, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    cluster_loc = os.path.join(processed_dir, 'ClusterCoordinates.csv')

    cluster_sz = parseArgs()
    cluster_df = pd.read_csv(cluster_loc)
    supercluster_df = generateSuperClusters(cluster_df, cluster_sz=cluster_sz)
    supercluster_csv_name = 'SuperClusterAssignments_Size{}.csv'.format(cluster_sz)
    supercluster_df.to_csv(os.path.join(processed_dir, supercluster_csv_name), index=False)

    supercluster_md_df = generateSuperClusterMetaData(supercluster_df)
    supercluster_csv_name = 'SuperClusters_Size{}.csv'.format(cluster_sz)
    supercluster_md_df.to_csv(os.path.join(processed_dir, supercluster_csv_name), index=False)

if __name__ == '__main__':
    main()
