"""
Util functions
"""

import numpy as np

def haversine_dist(this_lat, this_lon, lat_vec, lon_vec):
    """ Vectorized implementation of haversine  distance between
    a given scalar lat and lon to a list of lats and lons.
    Assumed given in standard coordinate forms.

    Parameters
    ----------
    this_lat : scalar
        Latitude of point for which we are computing distances
    this_lon : scalar
        Longitude of point for which we are computing distances
    lat_vec : np.array / scalar
        List of latitudes for which to compute distance against
    lon_vec : np.array / scalar
        List of longitudes

    Returns
    -------
    np.array
        List of distances from given point with same indexing as given lists.
    """
    this_lat, this_lon, lat_vec, lon_vec = map(np.radians, (this_lat, this_lon, lat_vec, lon_vec))
    dlat = lat_vec - this_lat
    dlon = lon_vec - this_lon
    a = np.square(np.sin(dlat/2)) + np.cos(this_lat) * np.multiply(np.cos(lat_vec),  np.square(np.sin(dlon/2)))
    c = 2 * np.arcsin(np.sqrt(a))
    return c

def haversine_dist_scale(dist_metric):
    if dist_metric is None:
        return 1
    metric_scales = {
        'km' : 6371
    }
    if dist_metric not in metric_scales.keys():
        raise('Invalid distance metric {}'.format(dist_metric))
    return metric_scales[dist_metric]

def compute_hav_dist_matrix(A, B, dist_metric=None):
    """Given two arrays of lat and lon vectors, calculates pointwise distances.
        Each row is a coordinate pair of lat and lon

    Parameters
    ----------
    A : np.array
        Array of lat long coords of size (a, 2)
    B : np.array
        Array of lat long coords of size (b, 2)
    dist_metric: string
        Distance metric to scale the return by

    Returns
    -------
    np.array
        returns point-wise haversine distances in earth radians
        scaled by magic number

    """
    # Check we're getting numpy arrays
    assert(type(A).__module__ == np.__name__ and type(B).__module__ == np.__name__)
    assert(A.shape[1] == 2 and B.shape[1] == 2)

    lenA, lenB = A.shape[0], B.shape[0]
    if lenB < lenA:
        # leads to faster run time so we iterate along the shorter matrix
        return compute_hav_dist_matrix(B, A).T

    def hav_dist_wrapper(this_coord):
        return haversine_dist(this_coord[0], this_coord[1], B[:,0], B[:,1])

    dists = np.apply_along_axis(hav_dist_wrapper, 1, A)

    return dists * haversine_dist_scale(dist_metric)


def testComputeDistMatrix():
    A = np.random.uniform(-360,360,(100,2))
    B = np.random.uniform(-360,360,(5000,2))
    C = compute_hav_dist_matrix(A,B)
    assert(C.shape == (100,5000))
    C_T = compute_hav_dist_matrix(B,A)
    assert(np.array_equal(C, C_T.T))
    assert(np.all(C >= 0))
    assert(np.all(C <= np.pi))

    import time
    def time_me(A, B):
        times = []
        for _ in range(100):
            start = time.time()
            compute_hav_dist_matrix(A,B)
            end = time.time()
            times.append(end-start)
        print('Average time taken: {}'.format(np.mean(times)))

    # testing the timing of swapping application order
    time_me(A,B)
    time_me(B,A)

def main():
    testComputeDistMatrix()

if __name__ == '__main__':
    main()
