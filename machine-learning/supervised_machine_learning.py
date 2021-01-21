# Create and Read Raster Catalog
import pandas as pd
from pyrasterframes.rasterfunctions import *

from pyrasterframes.utils import create_rf_spark_session

from pyspark import SparkFiles

from pyspark.sql.functions import lit

# We import various Spark components that we need to construct our Pipeline.
# These are the objects that will work in sequence to conduct the data preparation and modeling.
from pyrasterframes import TileExploder
from pyrasterframes.rf_types import NoDataFilter

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import os.path

import rasterio
from rasterio.plot import show, show_hist

# import os

os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

gt_aws_no_sign = '-Dgeotrellis.raster.gdal.option.AWS_NO_SIGN_REQUEST=YES'

spark = create_rf_spark_session(**{
    #     'spark.driver.extraJavaOptions': gt_aws_no_sign,
    #     'spark.executor.extraJavaOptions': gt_aws_no_sign
})

# The first step is to create a Spark DataFrame containing our imagery data.
# To achieve that we will create a catalog DataFrame.
# In the catalog, each row represents a distinct area and time, and each column is the URI to a band’s image product.
# In this example our catalog just has one row.
# After reading the catalog, the resulting Spark DataFrame may have many rows per URI, with a column corresponding to each band.

# The imagery for feature data will come from eleven bands of 60 meter resolution Sentinel-2 imagery.
# We also will use the scene classification (SCL) data to identify high quality, non-cloudy pixels.
uri_base = 's3://s22s-test-geotiffs/luray_snp/{}.tif'
# uri_base = 'file:///home/jenniferwu/Raster_Data_Set/s22s-test-geotiffs/luray_snp/{}.tif'
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12']
cols = ['SCL'] + bands

catalog_df = pd.DataFrame([
    {b: uri_base.format(b) for b in cols}
])

tile_size = 256
df = spark.read.raster(catalog_df, catalog_col_names=cols, tile_dimensions=(tile_size, tile_size)) \
    .repartition(100)

df = df.select(
    rf_crs(df.B01).alias('crs'),
    rf_extent(df.B01).alias('extent'),
    rf_tile(df.SCL).alias('scl'),
    rf_tile(df.B01).alias('B01'),
    rf_tile(df.B02).alias('B02'),
    rf_tile(df.B03).alias('B03'),
    rf_tile(df.B04).alias('B04'),
    rf_tile(df.B05).alias('B05'),
    rf_tile(df.B06).alias('B06'),
    rf_tile(df.B07).alias('B07'),
    rf_tile(df.B08).alias('B08'),
    rf_tile(df.B09).alias('B09'),
    rf_tile(df.B11).alias('B11'),
    rf_tile(df.B12).alias('B12'),
)
df.printSchema()

# Data Prep
# Label Data
# The land classification labels are based on a small set of hand drawn polygons in the GeoJSON file here.
# The property id indicates the type of land cover in each area.
# For these integer values, 1 is forest, 2 is cropland, and 3 is developed areas.

# We will create a very small Spark DataFrame of the label shapes and then join it to the raster DataFrame.
# Such joins are typically expensive, but in this case both datasets are quite small.
# To speed up the join for the small vector DataFrame, we put the broadcast hint on it, which will tell Spark to put a copy of it on each Spark executor.

# After the raster and vector data are joined, we will convert the vector shapes into tiles using the rf_rasterize function.
# This procedure is sometimes called “burning in” a geometry into a raster.
# The values in the resulting tile cells are the id property of the GeoJSON, which we will use as labels in our supervised learning task.
# In areas where the geometry does not intersect, the cells will contain NoData.
crses = df.select('crs.crsProj4').distinct().collect()
print('Found ', len(crses), 'distinct CRS.')
crs = crses[0][0]

spark.sparkContext.addFile(
    'https://github.com/locationtech/rasterframes/raw/develop/pyrasterframes/src/test/resources/luray-labels.geojson')
# spark.sparkContext.addFile('/home/jenniferwu/Raster_Data_Set/s22s-test-geotiffs/luray_snp/luray-labels.geojson')

label_df = spark.read.geojson(SparkFiles.get('luray-labels.geojson')) \
    .select('id', st_reproject('geometry', lit('EPSG:4326'), lit(crs)).alias('geometry')) \
    .hint('broadcast')

df_joined = df.join(label_df, st_intersects(st_geometry('extent'), 'geometry')) \
    .withColumn('dims', rf_dimensions('B01'))

# Add a column from ".tiff"
df_labeled = df_joined.withColumn('label',
                                  rf_rasterize('geometry', st_geometry('extent'), 'id', 'dims.cols', 'dims.rows')
                                  )

# Masking Poor Quality Cells
# To filter only for good quality pixels, we follow roughly the same procedure as demonstrated in the quality masking section of the chapter on NoData.
# Instead of actually setting NoData values in the unwanted cells of any of the imagery bands, we will just filter out the mask cell values later in the process.
df_labeled = df_labeled \
    .withColumn('mask', rf_local_is_in('scl', [0, 1, 8, 9, 10]))

# at this point the mask contains 0 for good cells and 1 for defect, etc
# convert cell type and set value 1 to NoData
df_mask = df_labeled.withColumn('mask',
                                rf_with_no_data(rf_convert_cell_type('mask', 'uint8'), 1.0)
                                )

df_mask.printSchema()

# Create ML Pipeline

# SparkML requires that each observation be in its own row, and those observations be packed into a single Vector object.
# The first step is to “explode” the tiles into a single row per cell or pixel with the TileExploder (see also rf_explode_tiles).
# If a tile cell contains a NoData it will become a null value after the exploder stage.
# Then we use the NoDataFilter to filter out any rows that missing or null values, which will cause an error during training.
# Finally we use the SparkML VectorAssembler to create that Vector.

# Recall above we set undesirable pixels to NoData, so the NoDataFilter will remove them at this stage.
# We apply the filter to the mask column and the label column, the latter being used during training.
# When it is time to score the model, the pipeline will ignore the fact that there is no label column on the input DataFrame.
exploder = TileExploder()

noDataFilter = NoDataFilter() \
    .setInputCols(['label', 'mask'])

assembler = VectorAssembler() \
    .setInputCols(bands) \
    .setOutputCol("features")

# We are going to use a decision tree for classification.
# You can swap out one of the other multi-class classification algorithms if you like.
# With the algorithm selected we can assemble our modeling pipeline.
classifier = DecisionTreeClassifier() \
    .setLabelCol('label') \
    .setFeaturesCol(assembler.getOutputCol())

pipeline = Pipeline() \
    .setStages([exploder, noDataFilter, assembler, classifier])

pipeline.getStages()

# Train the Model
# The next step is to actually run each step of the Pipeline we created, including fitting the decision tree model.
# We filter the DataFrame for only tiles intersecting the label raster because the label shapes are relatively sparse over the imagery.
# It would be logically equivalent to either include or exclude thi step, but it is more efficient to filter because it will mean less data going into the pipeline.
model_input = df_mask.filter(rf_tile_sum('label') > 0).cache()
model = pipeline.fit(model_input)

# Model Evaluation
# To view the model’s performance, we first call the pipeline’s transform method on the training dataset.
# This transformed dataset will have the model’s prediction included in each row.
# We next construct an evaluator and pass it the transformed dataset to easily compute the performance metric.
# We can also create custom metrics using a variety of DataFrame or SQL transformations.
prediction_df = model.transform(df_mask) \
    .drop(assembler.getOutputCol()).cache()
prediction_df.printSchema()

eval = MulticlassClassificationEvaluator(
    predictionCol=classifier.getPredictionCol(),
    labelCol=classifier.getLabelCol(),
    metricName='accuracy'
)

accuracy = eval.evaluate(prediction_df)
print("\nAccuracy:", accuracy)

# As an example of using the flexibility provided by DataFrames,
# the code below computes and displays the confusion matrix.
# cnf_mtrx = prediction_df.groupBy(classifier.getPredictionCol()) \
#     .pivot(classifier.getLabelCol()) \
#     .count() \
#     .sort(classifier.getPredictionCol())
# cnf_mtrx

# Visualize Prediction
# Because the pipeline included a TileExploder, we will recreate the tiled data structure.
# The explosion transformation includes metadata enabling us to recreate the tiles.
# See the rf_assemble_tile function documentation for more details.
# In this case, the pipeline is scoring on all areas, regardless of whether they intersect the label polygons.
# This is simply done by removing the label column, as discussed above.
# scored = model.transform(df_mask.drop('label'))
#
# retiled = scored \
#     .groupBy('extent', 'crs') \
#     .agg(
#     rf_assemble_tile('column_index', 'row_index', 'prediction', tile_size, tile_size).alias('prediction'),
#     rf_assemble_tile('column_index', 'row_index', 'B04', tile_size, tile_size).alias('red'),
#     rf_assemble_tile('column_index', 'row_index', 'B03', tile_size, tile_size).alias('grn'),
#     rf_assemble_tile('column_index', 'row_index', 'B02', tile_size, tile_size).alias('blu')
# )
# retiled.printSchema()

# Take a look at a sample of the resulting prediction and the corresponding area’s red-green-blue composite image.
# Note that because each prediction tile is rendered independently, the colors may not have the same meaning across rows.
# scaling_quantiles = retiled.agg(
#     rf_agg_approx_quantiles('red', [0.03, 0.97]).alias('red_q'),
#     rf_agg_approx_quantiles('grn', [0.03, 0.97]).alias('grn_q'),
#     rf_agg_approx_quantiles('blu', [0.03, 0.97]).alias('blu_q')
# ).first()
#
# retiled.select(
#     rf_render_png(
#         rf_local_clamp('red', *scaling_quantiles['red_q']).alias('red'),
#         rf_local_clamp('grn', *scaling_quantiles['grn_q']).alias('grn'),
#         rf_local_clamp('blu', *scaling_quantiles['blu_q']).alias('blu')
#     ).alias('tci'),
#     rf_render_color_ramp_png('prediction', 'ClassificationBoldLandUse').alias('prediction')
# )

# Writing Raster Data - GeoTIFFs
outfile = os.path.join('/tmp', 'geotiff-supervised-machine-learning.tif')

model.transform(df) \
    .groupBy('extent', 'crs') \
    .agg(
    rf_assemble_tile('column_index', 'row_index', 'prediction', tile_size, tile_size).alias('prediction'),
    rf_assemble_tile('column_index', 'row_index', 'B04', tile_size, tile_size).alias('red'),
    rf_assemble_tile('column_index', 'row_index', 'B03', tile_size, tile_size).alias('grn'),
    rf_assemble_tile('column_index', 'row_index', 'B02', tile_size, tile_size).alias('blu')
) \
    .write.geotiff(outfile, crs=crs, raster_dimensions=(1830 // 4, 1830 // 4))

# We can view the written file with `rasterio`
with rasterio.open(outfile) as src:
    for b in range(1, src.count + 1):
        print("Tags on band", b, src.tags(b))
    # View raster
    show(src)
