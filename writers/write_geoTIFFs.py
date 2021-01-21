# Writing Raster Data - GeoTIFFs
from pyrasterframes.utils import create_rf_spark_session

import os.path

import rasterio
from rasterio.plot import show, show_hist

# This will be our setup for the following examples
spark = create_rf_spark_session(**{
    'spark.driver.memory': '4G',
    'spark.ui.enabled': 'false'
})


def scene(band):
    b = str(band).zfill(2)  # converts int 2 to '02'
    print(b)
    print('2019072203257_B{}.TIF'.format(b))
    return '2019072203257_B{}.TIF'.format(b)


rf = spark.read.raster(scene(2), tile_dimensions=(256, 256))
rf.printSchema()

# GeoTIFF is one of the most common file formats for spatial data, providing flexibility in
# data encoding, representation, and storage.
# RasterFrames provides a specialized Spark DataFrame writer for rendering a RasterFrame to a GeoTIFF.
# It is accessed by calling `dataframe.write.geotiff`.
# Limitations and mitigations
# One downside to GeoTIFF is that it is <b><u>not</u></b> a big-data native format.
# To create a GeoTIFF, all the data to be written must be `collect`ed in the memory of the Spark driver.
# This means you must actively limit the size of the data to be written.
# It is trivial to lazily read a set of inputs that cannot feasibly be written to GeoTIFF in the same environment.
#
# When writing GeoTIFFs in RasterFrames, you should limit the size of the collected data.
# Consider filtering the dataframe by time or @ref:[spatial filters]
# (vector-data.md#geomesa-functions-and-spatial-relations).
#
# You can also specify the dimensions of the GeoTIFF file to be written using
# the `raster_dimensions` parameter as described below.
#
# ### Parameters
#
# If there are many _tile_ or projected raster columns in the DataFrame,
# the GeoTIFF writer will write each one as a separate band in the file.
# Each band in the output will be tagged the input column names for reference.
#
# * `path`: the path local to the driver where the file will be written
# * `crs`: the PROJ4 string of the CRS the GeoTIFF is to be written in
# * `raster_dimensions`: optional, a tuple of two ints giving the size of the resulting file.
# If specified, RasterFrames will downsample the data in distributed fashion using bilinear resampling.
# If not specified, the default is to write the dataframe at full resolution, which can result in an `OutOfMemoryError`.
#
# ### Example
#
# See also the example in the @ref:[unsupervised learning page](unsupervised-learning.md).
#
# Let's render an overview of a scene's red band as a small raster,
# reprojecting it to latitude and longitude coordinates
# on the [WGS84](https://en.wikipedia.org/wiki/World_Geodetic_System) reference ellipsoid
# (aka [EPSG:4326](https://spatialreference.org/ref/epsg/4326/)).
# Attempting to write a full resolution GeoTIFF constructed from multiple scenes is
# likely to result in an out of memory error.
# Consider filtering the dataframe more aggressively and using a smaller value for the `raster_dimensions` parameter.

outfile = os.path.join('/tmp', 'geotiff-overview.tif')
rf.write.geotiff(outfile, crs='EPSG:4326', raster_dimensions=(256, 256))

# We can view the written file with `rasterio`
with rasterio.open(outfile) as src:
    # View raster
    show(src, adjust='linear')
    # View data distribution
    show_hist(src, bins=50, lw=0.0, stacked=False, alpha=0.6,
              histtype='stepfilled', title="Overview Histogram")
