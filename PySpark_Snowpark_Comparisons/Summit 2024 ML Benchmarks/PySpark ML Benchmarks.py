# Databricks notebook source
!pip install xgboost

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import OneHotEncoder
from xgboost.spark import SparkXGBRegressor

import re
from getpass import getpass

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# admin = getpass()

# COMMAND ----------

password = getpass()

# COMMAND ----------

config = {
    'account': 'sfsenorthamerica-skhara',
    'user': 'admin',
    'password': password,
    'role': 'ACCOUNTADMIN',
    'warehouse': 'SSK_RESEARCH',
    'database': 'DEMO_DB',
    'schema': 'PUBLIC'
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### PySpark Session

# COMMAND ----------

from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import regexp_replace, col
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sfOptions = {
    "sfURL": "vkb96626.snowflakecomputing.com",
    "sfUser": config['user'],
    "sfPassword": config['password'],
    "sfDatabase": config['database'],
    "sfSchema": config['schema'],
    "sfWarehouse": config['warehouse'],
    "usestagingtable": 'ON'
}

SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"

# COMMAND ----------

spark = SparkSession.builder.appName('Basics').getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing -> Tranfer + Storage

# COMMAND ----------

# MAGIC %md
# MAGIC Load processed data from Snowflake and saving to DBX delta tables so that we can compare like to like.

# COMMAND ----------

df_raw = (spark.read.format(SNOWFLAKE_SOURCE_NAME)
        .options(**sfOptions)
        .option(
            "query",
            """select * from TPCDS_XGBOOST.DEMO.FEATURE_STORE_100;""",
        ).load())

df_raw.write.mode('overwrite').saveAsTable("FEATURE_STORE_100")

# COMMAND ----------

# MAGIC %md
# MAGIC # START

# COMMAND ----------

def step1_model_preprocess():
    feature_df = spark.read.format('delta').load('dbfs:/user/hive/warehouse/feature_store_100')
    snowdf = feature_df.drop(*['CA_ZIP','CUSTOMER_SK', 'C_CURRENT_HDEMO_SK', 'C_CURRENT_ADDR_SK', 'C_CUSTOMER_ID', 'CA_ADDRESS_SK', 'CD_DEMO_SK'])

    cat_cols = ['CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS']
    num_cols = ['C_BIRTH_YEAR', 'CD_DEP_COUNT']

    my_imputer = Imputer(
        inputCols=num_cols,
        outputCols=num_cols,  # Optionally, specify new output column names
        strategy="median"
    )
    df_prepared = my_imputer.fit(snowdf).transform(snowdf)

    # OHE of Categorical Cols
    # Step 1: Indexing categorical columns using StringIndexer
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed").setHandleInvalid("keep") for col in cat_cols]

    # Step 2: Encoding indexed columns using OneHotEncoder
    encoder = OneHotEncoder(
        inputCols=[indexer.getOutputCol() for indexer in indexers],
        outputCols=[col + "_ohe" for col in cat_cols]
    )

    # Step 3: Assembling all transformations into a Pipeline
    pipeline = Pipeline(stages=indexers + [encoder])

    # Step 4: Fitting the pipeline and transforming the DataFrame
    df_prepared = pipeline.fit(df_prepared).transform(df_prepared)

    # Cleaning column names to make it easier for future referencing
    cols = df_prepared.columns
    for old_col in cols:
        new_col = re.sub(r'[^a-zA-Z0-9_]', '', old_col)
        new_col = new_col.upper()
        df_prepared = df_prepared.withColumnRenamed(old_col, new_col)
    
    df_prepared = df_prepared.drop(*cat_cols)

    feature_cols = df_prepared.columns
    target_col = 'TOTAL_SALES'
    feature_cols.remove(target_col)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_prepared = assembler.transform(df_prepared)

    # Select the features and target variable
    df_prepared = df_prepared.select(col("features"), col(target_col).alias("label"))

    # Save the train and test sets as time stamped tables in Snowflake
    df_train, df_test = df_prepared.randomSplit([0.8, 0.2], seed=82)
    df_train.fillna(0).write.mode('overwrite').saveAsTable("TPC_TRAIN_XGB")
    df_test.fillna(0).write.mode('overwrite').saveAsTable("TPC_TEST_XGB")

    return df_prepared.count()

# COMMAND ----------

def step2_model_train():
    df_train = spark.read.format('delta').load('dbfs:/user/hive/warehouse/tpc_train_xgb')
    xgb_regressor = SparkXGBRegressor(
        features_col="features", 
        label_col="label",
        prediction_col="prediction",
        objective="reg:squarederror",
        numRound=100,
        maxDepth=6,
        eta=0.1,
        seed=123,
        numWorkers=32
    )

    model = xgb_regressor.fit(df_train)
    return model

# COMMAND ----------

def step3_model_inference(model):
    df_test = spark.read.format('delta').load('dbfs:/user/hive/warehouse/tpc_test_xgb')
    print(df_test.count())
    predictions = model.transform(df_test)
    print(predictions.count())
    return predictions

# COMMAND ----------

# Step 1: data preprocess
step1_model_preprocess()

# COMMAND ----------

# Step 2: model training
model = step2_model_train()

# COMMAND ----------

# Step 3: model inference
results = step3_model_inference(model)

# COMMAND ----------



# COMMAND ----------


