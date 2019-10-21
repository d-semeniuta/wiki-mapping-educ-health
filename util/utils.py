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
