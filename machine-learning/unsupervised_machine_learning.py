# Imports and Data Preparation
# We import various Spark components needed to construct our Pipeline.
import pandas as pd
from pyrasterframes import TileExploder
from pyrasterframes.rasterfunctions import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

from pyrasterframes.utils import create_rf_spark_session

from pyrasterframes.rf_types import CellType
# from pyrasterframes.rf_types import Extent

import os.path

import rasterio
from rasterio.plot import show, show_hist

spark = create_rf_spark_session()

# The first step is to create a Spark DataFrame of our imagery data.
# To achieve that we will create a catalog DataFrame using the pattern from the I/O page.
# In the catalog, each row represents a distinct area and time, and each column is the URI to a band’s image product.
# The resulting Spark DataFrame may have many rows per URI, with a column corresponding to each band.
filenamePattern = "https://github.com/locationtech/rasterframes/" \
                  "raw/develop/core/src/test/resources/L8-B{}-Elkton-VA.tiff"
catalog_df = pd.DataFrame([
    {'b' + str(b): filenamePattern.format(b) for b in range(1, 8)}
])

tile_size = 256
df = spark.read.raster(catalog_df, catalog_col_names=catalog_df.columns, tile_size=tile_size)
df = df.withColumn('crs', rf_crs(df.b1)) \
    .withColumn('extent', rf_extent(df.b1))
df.printSchema()

# In this small example, all the images in our `catalog_df` have the same @ref:[CRS](concepts.md
# #coordinate-reference-system-crs-), which we verify in the code snippet below.
# The `crs` object will be useful for visualization later.
crses = df.select('crs.crsProj4').distinct().collect()
print('Found ', len(crses), 'distinct CRS: ', crses)
assert len(crses) == 1
crs = crses[0]['crsProj4']

# Create ML Pipeline
# SparkML requires that each observation be in its own row,
# and features for each observation be packed into a single Vector.
# For this unsupervised learning problem, we will treat each pixel as an observation and each band as a feature.
# The first step is to “explode” the tiles into a single row per pixel.
# In RasterFrames, generally a pixel is called a cell.
exploder = TileExploder()

# To "vectorize" the band columns, we use the SparkML `VectorAssembler`.
# Each of the seven bands is a different feature.
# assembler = VectorAssembler() \
#     .setInputCols(list(catalog_df.columns)) \
#     .setOutputCol("features")

assembler = VectorAssembler(inputCols=list(catalog_df.columns), outputCol="features", handleInvalid="skip")

# For this problem, we will use the K-means clustering algorithm and configure our model to have 5 clusters.
kmeans = KMeans().setK(5).setFeaturesCol('features')

# We can combine the above stages into a single Pipeline.
pipeline = Pipeline().setStages([exploder, assembler, kmeans])

# Fit the Model and Score
# Fitting the pipeline actually executes exploding the tiles, assembling the features vectors,
# and fitting the K-means clustering model.
model = pipeline.fit(df)

# We can use the transform function to score the training data in the fitted pipeline model.
# This will add a column called prediction with the closest cluster identifier.
clustered = model.transform(df)

# Now let’s take a look at some sample output.
clustered.select('prediction', 'extent', 'column_index', 'row_index', 'features')

# If we want to inspect the model statistics,
# the SparkML API requires us to go through this unfortunate contortion to access the clustering results:
cluster_stage = model.stages[2]

# We can then compute the sum of squared distances of points to their nearest center,
# which is elemental to most cluster quality metrics.
metric = cluster_stage.computeCost(clustered)
print("Within set sum of squared errors: %s" % metric)

# Visualize Prediction
# We can recreate the tiled data structure using the metadata added by the TileExploder pipeline stage.

retiled = clustered.groupBy('extent', 'crs') \
    .agg(
    rf_assemble_tile('column_index', 'row_index', 'prediction',
                     tile_size, tile_size, CellType.int8())
)

# The resulting output is shown below.

# aoi = Extent.from_row(
#     retiled.agg(rf_agg_reprojected_extent('extent', 'crs', 'epsg:3857')) \
#         .first()[0]
# )
#
# retiled.select(rf_agg_overview_raster('prediction', 558, 507, aoi, 'extent', 'crs'))

# For comparison, the true color composite of the original data.
#  this is really dark
# df.select(rf_render_png('b4', 'b3', 'b2'))

# =========================================Writing Raster Data - write-GeoTIFFs=========================================
# Next we will @ref:[write the output to a GeoTiff file](raster-write.md#geotiffs).
# Doing so in this case works quickly and well for a few specific reasons that may not hold in all cases.
# We can write the data at full resolution, by omitting the `raster_dimensions` argument,
# because we know the input raster dimensions are small.
# Also, the data is all in a single CRS, as we demonstrated above.
# Because the `catalog_df` is only a single row, we know the output GeoTIFF value at a given location corresponds to
# a single input. Finally, the `retiled` `DataFrame` only has a single `Tile` column,
# so the band interpretation is trivial.
output_tif = os.path.join('/tmp', 'geotiff-unsupervised-machine-learning.tif')

retiled.write.geotiff(output_tif, crs=crs)

# ======================================We can view the written file with `rasterio`====================================
with rasterio.open(output_tif) as src:
    for b in range(1, src.count + 1):
        print("Tags on band", b, src.tags(b))
    # View raster
    show(src)
