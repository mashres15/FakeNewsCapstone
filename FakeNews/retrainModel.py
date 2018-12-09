#-------------------------------------
#***************Imports***************
#-------------------------------------
import pandas as pd
import multiprocessing #concurrency
import nltk
import gensim
import re

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
from urllib.parse import urlparse

import time

from textblob import TextBlob
import dill as pickle
# import pickle

from customTransfomers import *

from sqlalchemy import create_engine

engine = create_engine('postgresql://localhost/fakenews')

t0 = time.time()
#-------------------------------------
#***********Setting NTLK**************
#-------------------------------------

nltk.download('punkt')
nltk.download('stopwords')

#-------------------------------------
#***********Reading Data**************
#-------------------------------------
print("Reading Data ......")
df = pd.read_sql_query('SELECT * FROM "Fakenewscorpus"', con=engine)
df.rename(columns={'prediction':'type'}, inplace=True)
print("Cleaning the Data ......")

# Drop unused columns
df = df.drop("entry_date", axis = 1)

# Creating Feature from existing data
df['comment_plugin_count'] = 0
df['domain'] = df['url'].apply(lambda url : urlparse(url).netloc)

print("Filtering Data ......")
# If title is missing drop the entry
df = df.dropna(subset=['title', 'content'])

# Change missing authors to Anonymous
df.loc[df.authors.isnull(), 'authors'] = 'Anonymous'

# Get df where content is not null
df = df[df.content.notnull()]

print("Length of df: ", len(df))

#--------------------------------------------
#****Initialize encoders & feature hasher****
#--------------------------------------------

print("Initialize encoders ......")
# fh = FeatureHasher(n_features=10, input_type='string')
# le = LabelEncoder()
# le.fit(df.type.tolist())
# labels = le.transform(df.type.tolist())
#
# print("Get Params\n")
# print(list(le.inverse_transform([0,1])))

labels  = df['type'].apply(lambda x:  1 if x == 'fake' else 0)

#************** Word tokenizer **************
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()



#------------------------------------------------
#************ Implementing Pipelines ************
#------------------------------------------------
# print("Creating Pipeline ......")
text_data = ['content', 'title']
text_content = 'content'
numeric_data = ['reaction_count', 'comment_count', 'share_count', 'comment_plugin_count', 'page_rank_integer','rank']
hashing_data = ['domain', 'authors']

debug = False

hashing_pipeline = Pipeline([
                    ('selector', GetRespectiveData(hashing_data, value=True)),
                    ('hashing', FeatureHasher(n_features=20, input_type='string')),
                    ('Debug', DebugTransformer('Hash'))
                    ])

numeric_pipeline = Pipeline([
                    ('selector', GetRespectiveData(numeric_data)),
                    ('normalize', StandardScaler()),
                    ('Debug', DebugTransformer('Numeric', debug)
                    )
                    ])

text_pipeline = Pipeline([
                    ('selector', GetRespectiveData(text_data)),
                    ('feature_engineering', TextProcessor()),
                    ('Debug', DebugTransformer('text', debug))
                    ])

sentiment_pipeline = Pipeline([
                    ('selector', GetRespectiveData(text_content)),
                    ('sentiment_engineering', SentimentTransformer()),
                    ('Debug', DebugTransformer('sentiment', debug)),
                    ('normalize', StandardScaler())
                    ])

pl = Pipeline([
            ('union', FeatureUnion([
                    ('hashing', hashing_pipeline),
                    ('numeric', numeric_pipeline),
                    ('text', text_pipeline),
                    ('sentiment', sentiment_pipeline)
                     ])
            ),
            # ('clf', DecisionTreeClassifier(random_state=0))
            ('clf', RandomForestClassifier(random_state=0))
            ])


X = df[['domain', 'content', 'title', 'authors',
       'page_rank_integer', 'rank', 'reaction_count', 'comment_count',
       'share_count', 'comment_plugin_count']]

if __name__ == '__main__':
	print("Splitting Dataset ......")

	X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.30, random_state=42)

	tPipe0 = time.time()
	print("Fitting Pipeline ......")
	pl.fit(X_train, y_train)
	tPipe1 = time.time()
	tPipe = tPipe1 - tPipe0

	tPredict0 = time.time()
	print("Predicting Data ......\n")
	predicted = pl.predict(X_test)
	tPredict1 = time.time()
	tPredict = tPredict1 - tPredict0
	indPredTime = tPredict/len(predicted)

	print("Length of df_test: ", len(X_test))
	print("Length of df_train: ", len(X_train))

	pickle.dump(pl, open("rfclf.pkl", "wb"))

	accuracy = metrics.accuracy_score(y_test, predicted)
	f1_score = metrics.f1_score(y_test, predicted, average="weighted")
	cm = metrics.confusion_matrix(y_test, predicted)

	tCV0 = time.time()
	cv = np.mean(cross_val_score(pl, X, labels, cv=3))
	tCV1 = time.time()
	tCV = tCV1 - tCV0
	t1 = time.time()

	total_runtime = t1-t0
	print("***********************")
	print("Total RunTime (sec): ", t1)
	print("Pipeline fitting time (sec): ", tPipe)
	print("Prediction time for the whole set: (sec)", tPredict)
	print("Avg Prediction time for one entry: (sec)", indPredTime)
	print("Time for 3-fold CV (sec): ", tCV)
	print()
	print("***********************")
	print("Scoring Algorithm")
	print("***********************")
	print("accuracy: ", accuracy)
	print("f1-score: ", f1_score)
	print("Confusion Matrix: \n", cm)

	print("Classification Report")
	print(classification_report(y_test, predicted))

	print("3-fold CV accuracy: ", cv)

	file = open("pipeline_output.txt", "w")
	file.write("Scoring Alogrithm")
	file.write("accuracy: "+ accuracy)
	file.write("f1-score: "+ f1_score)
# 	file.write("Confusion Matrix: \n", cm)
	file.write("Classification Report")
	file.write(classification_report(y_test, predicted))
	file.write("3-fold CV accuracy: "+ cv)


	file.write("***********************")
	file.write("Total RunTime (sec): "+ t1)
	file.write("Pipeline fitting time (sec): "+ tPipe)
	file.write("Prediction time for the whole set: (sec)"+ tPredict)
	file.write("Avg Prediction time for one entry: (sec)"+ indPredTime)
	file.write("Time for 3-fold CV (sec): "+ tCV)
