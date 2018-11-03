import facebook
import pprint
import pandas as pd
import requests
import urllib.parse
import json
import sys
from newspaper import Article
from urllib.parse import urlparse
import os

import pickle

from .customTransfomers import *

token = "314429049360741|IKMSqj_l0-0GQYkelsf5caXLhXc"
graph = facebook.GraphAPI(access_token=token, version="3.0")

# *********************************************************
# ***************** Acquire Facebook Data *****************
# *********************************************************
def encode(data):
    encoded_string = ""
    for key, values in data.items():
        for i, domain in enumerate(values):
            if encoded_string:
                encoded_string += '&'
            encoded_string += key + "%5B" + str(i) + "%5D=" + domain
    return encoded_string

def getMetaData(url, domain = True):
	post = graph.get_object(id=url, fields='engagement')
	post['engagement']

	reaction_count = post['engagement']['reaction_count']
	comment_count = post['engagement']['comment_count']
	share_count = post['engagement']['share_count']
	comment_plugin_count = post['engagement']['comment_plugin_count']

	if domain:
		formdata ={'domains': url}

		request_url = 'https://openpagerank.com/api/v1.0/getPageRank?' + encode(formdata)
		headers = {'API-OPR': 'gccgs8wo4k44cgwc4gogwowwk08404kks8o4w0o0'}

		datas = requests.get(request_url, headers=headers)
		datas = datas.json()

		for response in datas['response']:
			domain = response['domain']
			if response['status_code'] == 200:
				post['engagement']['page_rank_integer']= response['page_rank_integer']
				post['engagement']['rank'] = response['rank']

			else:
				post['engagement']['page_rank_integer']= 0
				post['engagement']['rank'] = sys.maxsize

	#     print(post['engagement'])
	return post['engagement']

# *********************************************************
# ******************** Acquire URL Data *******************
# *********************************************************
def getData(url):
	print ("Getting metadata from url...")

	urlData = urlparse(url)
	domain = urlData.scheme +"://"+ urlData.netloc

	metaData = getMetaData(domain)
	print("Metadata of domain received...")

	urlMetaData = getMetaData(url, domain = False)
	print("Metadata of url received...")

	print("Scraping website...")

	article = Article(url)
	article.download()
	article.parse()

	authors = ","
	authors = authors.join(article.authors)
	print(article.authors)
	print(authors)
	
	title = article.title
	content = article.text

	date = article.publish_date

	print("Finished scraping website...")
	jsonData = {'url': url,
        'content': content,
        'title': title,
        'authors': authors,
        'page_rank_integer': metaData['page_rank_integer'],
        'rank': metaData['rank'],
        'reaction_count': metaData['reaction_count'],
        'comment_count': metaData['comment_count'],
        'share_count': metaData['share_count'],
        'comment_plugin_count': metaData['comment_plugin_count']
       }

	data = {'domain': [domain],
        'content': [content],
        'title': [title],
        'authors': [authors],
        'page_rank_integer': [metaData['page_rank_integer']],
        'rank': [metaData['rank']],
        'reaction_count': [metaData['reaction_count']],
        'comment_count': [metaData['comment_count']],
        'share_count': [metaData['share_count']],
        'comment_plugin_count': [metaData['comment_plugin_count']]
       }


	return pd.DataFrame.from_dict(data), jsonData

def newsPrediction(url):
    df, data = getData(url)
    basedir = os.path.abspath(os.path.dirname(__file__))
    clf = pickle.load(open(basedir + "/rfclf.pkl", "rb"))
    prediction = clf.predict(df)
    data['prediction'] = prediction
    print(prediction)
    return data
