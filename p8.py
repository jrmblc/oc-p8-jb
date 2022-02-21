import sys
from io import BytesIO

import boto3
from PIL import Image

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

BUCKET = sys.argv[1]
IMAGES_ORIGIN_FOLDER = sys.argv[2]
FOLDER_CSV = sys.argv[3]
MAX_PIX = (int(sys.argv[4]),int(sys.argv[4])) #max image resize in pixels (width, height)

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
# print(subfolders)

# Download of the image data
rdd_empty = spark.sparkContext.emptyRDD()
structure_a = StructType([StructField('image',
                        StructType([StructField('origin', StringType(), True),
                                    StructField('height', IntegerType(), True),
                                    StructField('width', IntegerType(), True),
                                    StructField('nChannels', IntegerType(), True),
                                    StructField('mode', IntegerType(), True),
                                    StructField('data', BinaryType(), True)]), True)])
df = spark.createDataFrame(rdd_empty, structure_a)

for sf in subfolders:
    df_plus = spark.read.format('image').option('dropInvalid', True)\
                                        .load('s3://'+BUCKET \
                                               + IMAGES_ORIGIN_FOLDER + '/'\
                                               + sf)
    df = df.union(df_plus)

# Extraction of the labels
labeler = udf(lambda x: x.split('/')[-2], StringType())
df = df.withColumn('label', labeler(col('image.origin')))

# Image processing (resize)
def image_process(image, max_pix):
    img = Image.frombuffer('RGB',(image.width,image.height), bytes(image.data))
    img.thumbnail(max_pix)
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    
    image = [img.height, img.width, bytearray(buffer.getvalue())]
    return image

structure_b = StructType([StructField('height', IntegerType(), True),
                        StructField('width', IntegerType(), True),
                        StructField('data', BinaryType(), True)])

process = udf(lambda x: image_process(x, MAX_PIX), structure_b)
df = df.withColumn('image_processed', process(col('image')))

# df.printSchema()

# df.select('image.origin', 'image.height', 
#           'image.width', 'image.nChannels', 
#           'image.mode', 'image.data').show(5)

# df.select('image_processed.height', 'image_processed.width',
#           'image_processed.data').show(5)

# CSV backup
df_csv = df.select(col('label'), base64(col("image_processed.data"))\
                .alias('image (base64 encoded)'))
df_csv.write.format('csv').option('header','true')\
                .save('s3://' + BUCKET + FOLDER_CSV , mode='overwrite')

