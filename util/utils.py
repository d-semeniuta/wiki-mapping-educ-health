"""
Util functions
"""
import os

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

REPO_HEAD = os.path.abspath('../')

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
    # Radius of earth
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

def getCoordinateCentroid(coords):
    """ Get centroid given an array of coordinates

    Parameters
    ----------
    coords : np.array
        Array of lat long coords of size (a, 2)

    Returns
    -------
    tuple
        Lat and Lon of centroid of given coords

    """
    # Check we're getting numpy arrays
    assert(type(coords).__module__ == np.__name__)
    assert(coords.shape[1] == 2)
    coords = np.radians(coords)
    lat, lon = coords[:,0], coords[:,1]
    # compute location in 3D axis
    X = np.cos(lat) * np.cos(lon)
    Y = np.cos(lat) * np.sin(lon)
    Z = np.sin(lat)

    x, y, z = np.mean(X), np.mean(Y), np.mean(Z)
    centroid_lon = np.arctan2(y, x)
    hyp = np.sqrt(x*x + y*y)
    centroid_lat = np.arctan2(z, hyp)

    return np.degrees(centroid_lat), np.degrees(centroid_lon)

# def produce_geo_scatter_plot(df, title, img_name, countries=None, color_col=None,
#                     size_col=None, out_dir=None, img_format='svg'):
#     """ Produce geo plot of desired data
#     To install requirements:
#         conda install -c plotly plotly-orca psutil requests
#
#     Parameters
#     ----------
#     df : Pandas.DataFrame
#         DataFrame of desired data to plot. Must include lat and lon data
#     title : string
#         Title of plot
#     img_name : string
#         Filename to give image
#     countries : string or list of strings
#         countries which to plot data of
#     color_col : string
#         Column which decides color of markers
#     size_col : string
#         Column which decides size of markers
#     out_dir : string
#         Path where to save image
#     img_format : string
#         File extension of image
#     """
#     if color_col is None and size_col is None:
#         raise(TypeError('Require at least one column argument'))
#     marker_color = df[color_col] if color_col is not None else 1
#     marker_size = df[size_col] if size_col is not None else 2
#     if countries is not None:
#         df = df.loc[df.country==countries]
#     if img_format not in ['png', 'jpeg', 'webp', 'svg', 'pdf']:
#         raise(ValueError('Unsupported file extension'))
#
#     fig = go.Figure(data=go.Scattergeo(
#             lon = df['lon'],
#             lat = df['lat'],
#             mode = 'markers',
#             marker = dict(
#                 size = marker_size,
#                 reversescale = True,
#                 autocolorscale = False,
#                 colorscale = 'Blues',
#                 cmin = 0,
#                 color = marker_color,
#                 cmax = marker_color.max(),
#                 colorbar_title=color_col
#             )))
#     fig.update_layout(
#             title = title,
#             geo_scope = None
#         )
#     fig.write_image(os.path.join(out_dir, '{}.{}'.format(img_name, img_format)))

def plotSingle_plotly(ins, outs, r2, save_dir, title, task):
    trace1 = go.Scatter(
        x=ins,
        y=outs,
        mode='markers',
        name='Predictions'
    )
    max_val = 3 if task == 'mated' else 1
    xi = np.arange(0,max_val,0.01)
    trace2 = go.Scatter(
        x=xi,
        y=xi,
        mode='lines',
        name='Fit'
    )

    annotation = go.layout.Annotation(
        x=0.05,
        y=max_val-0.05,
        text='$R^2 = {:.3f}$'.format(r2),
        showarrow=False,
    )
    layout = go.Layout(
        title = title,
        xaxis_title = 'Ground Truth',
        yaxis_title = 'Predictions',
        annotations=[annotation]
    )
    fig=go.Figure(data=[trace1,trace2], layout=layout)

    save_loc = os.path.join(save_dir, '{}.{}'.format(task, format))
    fig.write_image(save_loc)

def plotSingle_plt(ins, outs, r2, save_dir, title, task):
    fig, ax = plt.subplots()
    ax.scatter(ins, outs)
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.text(0.05, 0.95, 'R^2: {:.3f}'.format(r2),
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    max_res = 1 if task == 'imr' else 3
    ax.set_xlim(0,max_res)
    ax.set_ylim(0,max_res)
    plt.title(title)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    for format in ['png', 'svg', 'eps']:
        # save in multiple formats, just to be safe
        save_loc = os.path.join(save_dir, '{}.{}'.format(task, format))
        plt.savefig(save_loc, format=format, dpi=1200)
    plt.close()

def plotSingle(ins, outs, r2, save_loc, title, task, use_plotly=False):
    if use_plotly:
        plotSingle_plotly(ins, outs, r2, save_loc, title, task)
    else:
        plotSingle_plt(ins, outs, r2, save_loc, title, task)

def plotPreds(ins, outs, r2s, plot_info, use_plotly=False):
    title_task = {
        'imr': 'IMR', 'mated': 'Maternal Education'
    }
    for task in ins.keys():
        this_in = ins[task]
        this_out = outs[task]
        r2 = r2s[task]
        save_dir = plot_info['save_dir']
        title = '{}, {}'.format(plot_info['title'], title_task[task])
        plotSingle(this_in, this_out, r2, save_dir, title, task, use_plotly)
