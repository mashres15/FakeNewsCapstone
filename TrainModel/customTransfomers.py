#-------------------------------------
#***************Imports***************
#-------------------------------------
import pandas as pd
import multiprocessing #concurrency
import gensim
import re
import nltk 

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from textblob import TextBlob

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#************** Word tokenizer **************
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


#-----------------------------------------------
#************ Transformer Pipelines ************
#-----------------------------------------------
class TextProcessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, text = 'content', title = 'title'):
        self.text = text
        self.title = title
        
    def fit(self, X, y=None, **fit_params):
        return self
    
    def train_doc2vec(self, row):
        doc = row[self.text]
        title = row[self.title]

        # Tokenize the doc into sentences -> ['Sent1', 'Sent2']
        sents = nltk.sent_tokenize(doc)

        # Tokenize each sentence to word -> [['Sent1_Word1', 'Sent1_Word2'], ['Sent2_Word1', 'Sent2_Word2']]
        words_list = [tokenizer.tokenize(sent) for sent in sents]

        # Use Stemming to convert words to their root form and remove stop word
        stem_words_with_no_StopWord = [porter.stem(word) for sent_token in words_list 
                                   for word in sent_token if word not in stop_words]
		
        return gensim.models.doc2vec.TaggedDocument([word for word in stem_words_with_no_StopWord], [title])
    
    def transform(self, X, **transform_params):
        docCorpus = X.apply(self.train_doc2vec, axis=1)
        min_word_count = 3
        num_workers = multiprocessing.cpu_count()
        context_size = 7
        downsample = 1e-3
        size = 32

        docVecModel = gensim.models.doc2vec.Doc2Vec(docCorpus, vector_size=size, 
                                                    workers=num_workers, min_count=min_word_count, 
                                                    window=context_size, sample=downsample)
        
        docVecModel.save('docModel')
        docVecModel = gensim.models.Doc2Vec.load('docModel')

        txtVec = [docVecModel.infer_vector(doc.words) for doc in docCorpus]
        
        print("Text vectors generated...")
        return np.array(txtVec)

class SentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        polarity = X.apply(lambda x: TextBlob(x).sentiment.polarity)
        subjectivity = X.apply(lambda x: TextBlob(x).sentiment.subjectivity)
        features = pd.concat([polarity, subjectivity], axis=1)
        print("Senitment Features generate...")
        return features.values

    def fit(self, X, y=None):
        return self
        
class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name, debug = False):
        self.name = name
        self.debug = debug

    def transform(self, X):
        if self.debug:
            print(self.name, 'got', X.shape)
            print(X)
        if self.name == "Numeric":
            print("Numeric features transformed and normalized...")
        return X

    def fit(self, X, y=None):
        return self
  
#-----------------------------------------------
#************ Transformer Pipelines ************
#-----------------------------------------------
class GetRespectiveData(BaseEstimator, TransformerMixin):
	def __init__(self, data, value= False):
		self.data = data
		self.value = value

	def transform(self, x):
		if self.value:
			return x[self.data].values

		else:
			return x[self.data]

	def fit(self, X, y=None):
		return self
