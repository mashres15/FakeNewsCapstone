import pandas as pd
import csv
import sys
from pyspark.sql import SparkSession

csv.field_size_limit(9223372036854775807)


# Creating SparkSession (pySpark)
spark = SparkSession.builder.getOrCreate()

# Read large dataset
df = spark.read.csv('/Users/mshrestha/Desktop/fn/news_cleaned_2018_02_13.csv', header=True)

from pyspark.sql import SparkSession
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame

spark = SparkSession.builder.getOrCreate()

# Read large dataset
df = spark.read.csv('news_cleaned_2018_02_13.csv', header=True)

# Selecting Data
df = df.select([c for c in df.columns if c in ['domain','type','url','content','title','authors']])
df = df.dropna()
fake = df.filter(df.type =='fake').rdd.takeSample(False, 100000)
satire = df.filter(df.type == 'satire').rdd.takeSample(False, 100000)
bias = df.filter(df.type == 'bias').rdd.takeSample(False, 100000)
conspiracy = df.filter(df.type == 'conspiracy').rdd.takeSample(False, 100000)
junksci = df.filter(df.type == 'junksci').rdd.takeSample(False, 100000)
hate = df.filter(df.type == 'hate').rdd.takeSample(False, 100000)
clickbait = df.filter(df.type == 'clickbait').rdd.takeSample(False, 100000)
unreliable = df.filter(df.type == 'unreliable').rdd.takeSample(False, 100000)
political = df.filter(df.type == 'political').rdd.takeSample(False, 100000)
reliable = df.filter(df.type == 'reliable').rdd.takeSample(False, 900000)


# Combining all kinds of data
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

ndf = unionAll(fake, satire, bias, conspiracy, junksci, clickbait, unreliable, political, reliable)

ndf.to_csv('newMixedCorpus.csv')

