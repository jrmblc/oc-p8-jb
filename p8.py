import sys
import io

import pandas as pd
import numpy as np
from PIL import Image
import boto3

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import PCA

BUCKET = sys.argv[1]
IMAGES_ORIGIN_FOLDER = sys.argv[2]
BACKUP_FOLDER = sys.argv[3]

spark = SparkSession.builder.appName('p8').getOrCreate()

s3 = boto3.resource('s3')
s3_bucket = s3.Bucket(BUCKET)

# Identification of the images subfolders (fruits names)
files = []
for obj in s3_bucket.objects.filter(Prefix=IMAGES_ORIGIN_FOLDER[1:]+'/'):
    files.append(obj.key)  

rdd_subfolders = spark.sparkContext.parallelize(files[1:]) # [1:] exclude the origin folder 
rdd_subfolders = rdd_subfolders.map(lambda x: (x.split('/')[-2],1))
rdd_subfolders = rdd_subfolders.reduceByKey(lambda x, y : x + y)
subfolders = rdd_subfolders.keys().collect()

# Download of the images data
rdd_empty = spark.sparkContext.emptyRDD()
structure_image = StructType([StructField('path', StringType(), True),
                              StructField('modificationTime', TimestampType(), True),
                              StructField('length', LongType(), True),
                              StructField('content', BinaryType(), True)])
images = spark.createDataFrame(rdd_empty, structure_image)

for sf in subfolders:
    images_plus = spark.read.format('binaryFile')\
                        .option('pathGlobFilter','*.jpg')\
                        .option('recursiveFileLookup','true')\
                        .load('s3://'+BUCKET \
                                     + IMAGES_ORIGIN_FOLDER + '/' + sf)
    images = images.union(images_plus)

# Preparation of the model
model = ResNet50(include_top=False)
bc_model_weights = spark.sparkContext.broadcast(model.get_weights())

def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  model = ResNet50(weights=None, include_top=False)
  model.set_weights(bc_model_weights.value)
  return model

# Define image loading and featurization logic in a Pandas UDF
def preprocess(content):
  """
  Preprocesses raw image bytes for prediction.
  """
  img = Image.open(io.BytesIO(content)).resize([224, 224])
  arr = img_to_array(img)
  return preprocess_input(arr)

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  input = np.stack(content_series.map(preprocess))
  preds = model.predict(input)
  # For some layers, output features will be multi-dimensional tensors.
  # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
  output = [p.flatten() for p in preds]
  return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
  # for multiple data batches.  This amortizes the overhead of loading big models.
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)

# Featurization to the DataFrame of images
# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# We can now run featurization on our entire Spark DataFrame.
# NOTE: This can take a long time (about 10 minutes) since it applies a large model to the full dataset.
features_df = images.repartition(16).select(col("path"), featurize_udf("content").alias("features"))

# Extracting labels from images
labeler = udf(lambda x: x.split('/')[-2], StringType())
features_df = features_df.withColumn('label', labeler(col('path')))

# Vectorization of image features for dimensional reduction
vectorizer = udf(lambda x: Vectors.dense(x), VectorUDT())
features_df = features_df.withColumn('features_vec', vectorizer(col('features')))

# Dimensional reduction
pca = PCA(k=30, inputCol='features_vec', outputCol = 'features_pca')
model = pca.fit(features_df)

result_df = model.transform(features_df).select('label','features_pca')

# Backup of the result dataframe
result_df.write.option('mergeSchema', 'true').parquet('s3://' + BUCKET + BACKUP_FOLDER, mode='overwrite')

