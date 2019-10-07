"""
Please do not remove these lines
Authored by George Azzari
A few small changes by Chris Yeh and Anthony Perez
And now Riley DeHaan
Taken from: https://github.com/ermongroup/WikipediaPovertyMapping.git
"""

# must install conda env with:
# conda create -n testgdal -c conda-forge gdal vs2015_runtime=14
# then must set environment variable with:
# setx PROJ_LIB "C:\Program Files\GDAL\projlib"

import gdal, osr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio

class GeoProps(object):
    def __init__(self):
        self.eDT = None
        self.Proj = None
        self.GeoTransf = None
        self.Driver = None
        self.Flag = False
        self.xOrigin = None
        self.yOrigin = None
        self.pixelWidth = None
        self.pixelHeight = None
        self.srs = None
        self.srsLatLon = None

    def import_geogdal(self, gdal_dataset):
        """
        adfGeoTransform[0] /* top left x */
        adfGeoTransform[1] /* w-e pixel resolution */
        adfGeoTransform[2] /* 0 */
        adfGeoTransform[3] /* top left y */
        adfGeoTransform[4] /* 0 */
        adfGeoTransform[5] /* n-s pixel resolution (negative value) */
        :param gdal_dataset: a gdal dataset
        :return: nothing, set geooproperties from input dataset.
        """
        gdal.SetCacheMax(2**30)
        self.eDT = gdal_dataset.GetRasterBand(1).DataType
        self.Proj = gdal_dataset.GetProjection()
        self.GeoTransf = gdal_dataset.GetGeoTransform()
        self.Driver = gdal.GetDriverByName("GTiff")
        self.xOrigin = self.GeoTransf[0]
        self.yOrigin = self.GeoTransf[3]
        self.pixelWidth = self.GeoTransf[1]
        self.pixelHeight = self.GeoTransf[5]
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(self.Proj)
        self.srsLatLon = self.srs.CloneGeogCS()
        self.Flag = True

    def get_affinecoord(self, geolon, geolat):
        """Returns coordinates in meters (affine) from degrees coordinates (georeferenced)"""
        ct = osr.CoordinateTransformation(self.srsLatLon, self.srs)
        tr = ct.TransformPoint(geolon, geolat)
        xlin = tr[0]
        ylin = tr[1]
        return xlin, ylin

    def get_georefcoord(self, xlin, ylin):
        """Returns coordinates in degrees (georeferenced) from coordinates in meters (affine)"""
        ct = osr.CoordinateTransformation(self.srs, self.srsLatLon)
        tr = ct.TransformPoint(xlin, ylin)
        geolon = tr[0]
        geolat = tr[1]
        return geolon, geolat

    def lonlat2colrow(self, lon, lat):
        """Returns the [col, row] of a pixel given its lon, lat coordinates
        NOTE: the lon/lat coordinates must have the same units (e.g. degrees or meters) as
        self.xOrigin, self.pixelWidth, etc.
        """
        col = int((lon - self.xOrigin) / self.pixelWidth)
        row = int((lat - self.yOrigin) / self.pixelHeight)
        # print "(long,lat) = (",GeoX, ",", GeoY,") --> (col,row) = (",xOffset,",",yOffset,")"
        # NOTE: watch out! if you're using this to read a 2D np.array, remember
        # that xOffset = col, yOffset = row
        return [col, row]

    def lonlat2colrow_batch(self, lon, lat):
        """Vectorized version of lonlat2colrow
        Args:
            lon: 1-D np.array of longitudes
            lat: 1-D np.array of latitudes (same length as lon)
        NOTE: the lon/lat coordinates must have the same units (e.g. degrees or meters) as
        self.xOrigin, self.pixelWidth, etc.
        Returns: tuple of (col, row) which are 1-D np.arrays of ints
        """
        col = np.int_((lon - self.xOrigin) / self.pixelWidth)
        row = np.int_((lat - self.yOrigin) / self.pixelHeight)
        # Bug fix, if lon or lat is a 1-D array of shape (1,) then a bug will occur where the output is
        # an integer instead of an array
        if lon.shape[0] == 1:
            col = np.array([col])
        if lat.shape[0] == 1:
            row = np.array([row])
        return col, row

    def colrow2lonlat(self, col, row):
        """ Returns the (lon, lat) of a pixel given its (col, row)
        NOTE: the returned lon/lat coordinates will have the same units (e.g. degrees or
        meters) as self.xOrigin, self.pixelWidth, etc.
        """
        lon = col * self.pixelWidth + self.xOrigin
        lat = row * self.pixelHeight + self.yOrigin
        return [lon, lat]

    def get_center_coord(self, raster_array_shape, affine=False):
        """ Input: raster_array_shape is the output of gdalobject.np_array.shape, which is (#rows, #cols)
            Returns: coordinate (lon, lat) of the center of the raster."""
        s = raster_array_shape
        ul = self.colrow2lonlat(0, 0)
        lr = self.colrow2lonlat(s[1], s[0])
        lon_ext_m = lr[0] - ul[0]
        lat_ext_m = ul[1] - lr[1]
        lon_cntr = ul[0] + lon_ext_m / 2
        lat_cntr = lr[1] + lat_ext_m / 2
        if affine:
            return lon_cntr, lat_cntr
        if not affine:
            return self.get_georefcoord(lon_cntr, lat_cntr)

    def get_raster_extent(self, raster_array_shape):
        """ Input: raster_array_shape is the output of gdalobject.np_array.shape, which is (#rows, #cols)
            Returns: extent (in meters) of the raster."""
        s = raster_array_shape
        ul = self.colrow2lonlat(0, 0)
        lr = self.colrow2lonlat(s[1], s[0])
        lon_ext_m = lr[0] - ul[0]
        lat_ext_m = ul[1] - lr[1]
        return lon_ext_m, lat_ext_m

    def get_small_pxlwin(self, lon, lat, dpx, dpy):
        cen_col, cen_row = self.lonlat2colrow(lon, lat)
        rows = range(cen_row - dpy, cen_row + dpy + 1, 1)
        columns = range(cen_col - dpx, cen_col + dpx + 1, 1)
        row_indx = []
        col_indx = []
        for i in rows:
            row_indx.append(i)
        for j in columns:
            col_indx.append(j)
        return np.array(row_indx), np.array(col_indx)

    def get_img_from_coord_corners(self, UpLeft_lat, UpLeft_lon, LowerRight_lat, LowerRight_lon, raster):
            UpLeft_col, UpLeft_row = geo_prop.lonlat2colrow(UpLeft_lon, UpLeft_lat)
            LowerRight_col, LowerRight_row = geo_prop.lonlat2colrow(LowerRight_lon, LowerRight_lat)
            xsize, ysize = LowerRight_col - UpLeft_col, LowerRight_row - UpLeft_row

            # obtain numpy array from raster
            raster_img = raster.ReadAsArray(xoff=UpLeft_col, yoff=UpLeft_row, xsize=xsize, ysize=ysize, buf_obj=None,
                                            buf_xsize=None,
                                            buf_ysize=None, buf_type=None, callback=None, callback_data=None)
            return raster_img

    def get_coord_centered_img(self, lat, lon, NS_length, EW_length, raster, save=False, filename=None):
        # produces image of a NS_length km by EW_length km square as a numpy array centered on (lat, lon) coordinates

        # radius of Earth: 6371 km on average (Google)
        # 6371 km * 2*pi radians/360 deg = 111.1949 km/deg of latitude shift,
        # cos(latitude)*111.1949 km/deg of longitude shift for approximate spherical conversion
        upper_left_lat = lat + NS_length/(2*111.1949)
        upper_left_long = lon - EW_length/(2*111.1949*np.cos(upper_left_lat*np.pi/180))
        lower_right_lat = lat - NS_length/(2*111.1949)
        lower_right_long = lon + EW_length/(2*111.1949*np.cos(lower_right_lat*np.pi/180))
        raster_img = self.get_img_from_coord_corners(upper_left_lat, upper_left_long,
                                               lower_right_lat, lower_right_long, raster)

        # MORE ACCURATE USING SPHERICAL CONVERSION WITH COSINE(LAT), BUT THAT WILL MAKE THE RESULTING IMAGES HAVE DIFFERENT
        # SIZES...

        if save:
            if type(filename) is str:
                imageio.imwrite('../data/raw/' + filename, raster_img)

        return raster_img


class ImageComposite(object):

    def __init__(self, imgpath, nbands):

        self.imgpath = imgpath
        self.gdal_dataset = gdal.Open(imgpath)
        self.geoprops = GeoProps()
        self.geoprops.import_geogdal(self.gdal_dataset)
        self.bands = range(1, nbands)
        self.nbands = nbands

    def getpxwin(self, lon, lat, nrows, ncols, fpath, addgeo=True):

        rowindx, colindx = self.geoprops.get_small_pxlwin(lon, lat, ncols/2, nrows/2)
        # print("Max rowindx: ", rowindx.max())
        # print("Min rowindx: ", rowindx.min())
        # print("Max colindx: ", colindx.max())
        # print("Min colindx: ", colindx.min())
        ul = self.geoprops.colrow2lonlat(colindx.min(), rowindx.min())
        # print('GeoT: ', [ul[0], self.geoprops.pixelWidth, 0, ul[1], 0, self.geoprops.pixelHeight])
        # print('ul coords: ', ul)

        # One of Byte, UInt16, Int16, UInt32, Int32, Float32, Float64,
        # and the complex types CInt16, CInt32, CFloat32, and CFloat64.
        gdal_datatype = self.geoprops.eDT
        driver = self.geoprops.Driver
        dst_ds = driver.Create(fpath, ncols, nrows, self.nbands, gdal_datatype)

        if addgeo:
            dst_ds.SetGeoTransform([ul[0], self.geoprops.pixelWidth, 0, ul[1], 0, self.geoprops.pixelHeight])
            dst_ds.SetProjection(self.geoprops.Proj)

        # print(ul[0], ul[1], ncols, nrows)
        for i in self.bands:
            bandarray = self.gdal_dataset.GetRasterBand(i).ReadAsArray(colindx.min(), rowindx.min(), ncols, nrows)
            # print(np.all((bandarray == 0) + (bandarray == -32768)))
            # bandarray = bandarray.astype(np.float32)
            # print(np.all((bandarray == 0) + (bandarray == -32768)))
            dst_ds.GetRasterBand(i).WriteArray(bandarray)

        dst_ds = None

    def _pdrowfu(self, pdrow, lonindx, latindx, nrows, ncols, prefix, suffix, basepath, verbose=True):

        lon = pdrow[lonindx]
        lat = pdrow[latindx]

        if verbose:
            print('Tiling {0}x{1} image around (lat, lon)=({2}, {3})'.format(str(nrows), str(ncols),
                                                                             str(round(lat, 2)), str(round(lon, 2))))

        rname = 'ROW{0}_LON{1}_LAT{2}'.format(str(int(pdrow[0])), str(round(lon, 2)), str(round(lat, 2)))
        fname = "{0}_{1}_{2}.tif".format(prefix, rname, suffix)

        self.getpxwin(pdrow[lonindx], pdrow[latindx], nrows, ncols, "{0}/{1}".format(basepath, fname))

    def getgridwins(self, gridpath, lonindx, latindx, nrows, ncols, prefix, suffix, exportpath):

        p = pd.read_csv(gridpath)
        p.apply(self._pdrowfu, axis=1, raw=True, args=(lonindx, latindx, nrows, ncols, prefix, suffix, exportpath))

def deg_min_sec2deg_float(deg, min, sec):
    return float(deg) + min/60.0 + sec/3600.0

if __name__ == '__main__':
    # must install conda env with:
    # conda create -n testgdal -c conda-forge gdal vs2015_runtime=14
    # then set environment variable and restart IDE (Riley was using PyCharm):
    # setx PROJ_LIB "C:\Program Files\GDAL\projlib"

    # open GDAL data set
    filepath = r"../data/raw/GUF_Continent_Africa.tif"

    # for reduced image data set (better for seeing all of Africa at once and getting a sense of the coordinate system)
    # produces data set with each pixel corresponding to a tenth of a degree shift in lat/long on a side
    # import os
    # os.system('gdal_translate -r lanczos -tr 0.1 0.1  -co COMPRESS=LZW ../data/GUF_Continent_Africa.tif '
    #           '../data/raw/GUF_Continent_Africa_tenth.tif')

    # reduced_filepath = r"../data/raw/GUF_Continent_Africa_tenth.tif"

    # Open the file:
    raster = gdal.Open(filepath)

    if not raster:
        print("Raster file not loaded correctly.")

    # Check type of the variable 'raster'
    print(type(raster))
    # print(raster.GetProjection())

    # import GDAL data into GeoProps object for ease of use
    geo_prop = GeoProps()
    geo_prop.import_geogdal(raster)

    # get geocoordinates from affine coordinates (pixel coordinates? comments in code say in "meters" but I'm not sure
    # what that would be referencing in the standard)

    # # some example conversions from pixels to geocoordinates
    # Xpixel = 300.0
    # Yline = 300.0
    # Xgeo = geo_prop.GeoTransf[0] + Xpixel * geo_prop.GeoTransf[1] + Yline * geo_prop.GeoTransf[2]
    # Ygeo = geo_prop.GeoTransf[3] + Xpixel * geo_prop.GeoTransf[4] + Yline * geo_prop.GeoTransf[5]

    # obtain individual image centered around some geocoordinates as np.array
    # example: Upper-left corner in southern Tanzania, Madagascar shown prominently

    # UpLeft_lat, UpLeft_lon = -10.0, 37.0
    # UpLeft_col, UpLeft_row = geo_prop.lonlat2colrow(UpLeft_lon, UpLeft_lat)
    # # coordinates of the capital of Madagascar, can see lower-right corner just on the edge of the lights of the capital
    # LowerRight_lat, LowerRight_lon = -deg_min_sec2deg_float(18, 52, 3), deg_min_sec2deg_float(47, 31, 7)
    # LowerRight_col, LowerRight_row = geo_prop.lonlat2colrow(LowerRight_lon, LowerRight_lat)
    # xsize, ysize = LowerRight_col - UpLeft_col, LowerRight_row - UpLeft_row
    # # xsize, ysize = 20000, 20000
    #
    # # max sizes on reduced data set: xsize=889, ysize=722
    # # obtain numpy array from raster
    # raster_img = raster.ReadAsArray(xoff=UpLeft_col, yoff=UpLeft_row, xsize=xsize, ysize=ysize, buf_obj=None, buf_xsize=None,
    #                                 buf_ysize=None, buf_type=None, callback=None, callback_data=None)

    # obtain individual image centered around some geocoordinates as np.array
    # example: Upper-left corner in southern Tanzania, Madagascar shown prominently

    # lat, lon = -10.0, 37.0
    # col, row = geo_prop.lonlat2colrow(lon, lat)
    #
    # # max sizes on reduced data set: xsize=889, ysize=722
    # # obtain numpy array from raster
    # raster_img = raster.ReadAsArray(xoff=col, yoff=row, xsize=20000, ysize=20000, buf_obj=None, buf_xsize=None,
    #                                 buf_ysize=None, buf_type=None, callback=None, callback_data=None)

    raster_img = geo_prop.get_coord_centered_img(-18.88, 47.51, 50, 50, raster, save=True, filename='Antananarivo.jpg')

    # render image
    plt.imshow(raster_img, interpolation='nearest')
    plt.show()

