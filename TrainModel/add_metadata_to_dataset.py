import facebook
import pprint
import pandas as pd
import requests
import urllib.parse
import json
import sys
from urllib.parse import urlparse

token = "314429049360741|IKMSqj_l0-0GQYkelsf5caXLhXc"
# token = "280667649422059|xntRuFyVtSI9Cgx1cKFQYs9e7Gc"
graph = facebook.GraphAPI(access_token=token, version="3.0")
df = pd.read_csv('newMixedCorpus.csv')

# ----------------------------------------------------------
# Method to chuck data
# ----------------------------------------------------------
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# ----------------------------------------------------------
# Get all metadata
# ----------------------------------------------------------
def getMetaData(url):
    
    # Facebook Request
    post = graph.get_object(id=url, fields='engagement')
    post['engagement']
    
    reaction_count = post['engagement']['reaction_count']
    comment_count = post['engagement']['comment_count']
    share_count = post['engagement']['share_count']
    comment_plugin_count = post['engagement']['comment_plugin_count']
    
    formdata ={'domains': url}
    
    # open page rank
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

print(post['engagement'])
return post['engagement']

print(getMetaData('http://nytimes.com'))

# ----------------------------------------------------------
# Facebook API
# ----------------------------------------------------------
def getFacebookData(df, df_url, export=False, chunker_size = 50):
    urls_set = set(df[df_url].tolist())
    urls_list =[]
    for url in urls_set:
        urls_list.append('http://'+url)
    
    data = {}
    for urls in chunker(urls_list, chunker_size):
        post = graph.get_objects(ids=urls, fields='engagement')
        for url in post:
            data[url] = post[url]['engagement']
    
    facebookData = pd.DataFrame.from_dict(data, orient='index')
    
    if export: facebookData.to_csv('facebookData.csv')

facebookData.reset_index(level=0, inplace=True)
    
    return facebookData

# ----------------------------------------------------------
# Open Page Rank
# ----------------------------------------------------------
def encode(data):
    encoded_string = ""
    for key, values in data.items():
        for i, domain in enumerate(values):
            if encoded_string:
                encoded_string += '&'
            encoded_string += key + "%5B" + str(i) + "%5D=" + domain
return encoded_string

def getRankAPI(domains, chunker_size=100):
    ranking = {}
    
    for urls in chunker(domains, chunker_size):
        formdata ={'domains': urls}
        
        url = 'https://openpagerank.com/api/v1.0/getPageRank?' + encode(formdata)
        headers = {'API-OPR': 'gccgs8wo4k44cgwc4gogwowwk08404kks8o4w0o0'}
        
        datas = requests.get(url, headers=headers)
        datas = datas.json()
        
        for response in datas['response']:
            domain = response['domain']
            if response['status_code'] == 200:
                ranking[domain] = {'page_rank_integer': response['page_rank_integer'], 'rank': response['rank']}
            
            else:
                ranking[domain] = {'page_rank_integer': 0, 'rank': sys.maxsize}

ranks_df = pd.DataFrame.from_dict(ranking, orient='index')
ranks_df.reset_index(level=0, inplace=True)
ranks_df['index'] = "http://" + ranks_df['index']
    
    return ranks_df

def getMetaData(df, df_url='site_url', filename="Metadata.csv"):
    fb_df = getFacebookData(df, df_url)
    domains = fb_df['index'].tolist()
    ranks_df = getRankAPI(domains)
    
    df = pd.merge(ranks_df, fb_df, on="index")
    df.to_csv(filename)
    return df

# ----------------------------------------------------------
# Getting Data for FakeNewsCorpus
# ----------------------------------------------------------
df = pd.read_csv('newMixedCorpus.csv')
metaData = getMetaData(df, 'domain')
metaData['index'] = metaData['index'].str[7:]
fakecorupus = df.merge(metaData, left_on="domain", right_on="index")

# ----------------------------------------------------------
# Exporting FakeNewsCorpus with Metadata
# ----------------------------------------------------------
fakecorupus.to_csv("fakecorpusWithMeta.csv")


# ----------------------------------------------------------
# Getting Data for GRFN Dataset
# ----------------------------------------------------------
df = pd.read_csv('fake.csv')
df = df[df['language']== 'english']
df = df.drop(columns=['uuid', 'ord_in_thread', 'published', 'language', 'crawled', 'main_img_url',])
df.loc[df['title'] != df['thread_title'],'title'] = df[df['title'] != df['thread_title']]['thread_title']
df = df.dropna(subset=['title'])

df.loc[df.author.isnull(), 'author'] = 'Anonymous'
df = df[df.text.notnull()]
df = df[['author', 'title', 'text', 'site_url', 'type']]
df = df.rename(index=str, columns={"author": "authors", "text": "content", "site_url": "url"})

fakedf = getMetaData(df, 'url', 'GettingFakeNewswithMetadata')
fakedf['index'] = fakedf['index'].str[7:]

fakedf = df.merge(fakedf, left_on="url", right_on="index")

# ----------------------------------------------------------
# Exporting GRFN Dataset with Metadata
# ----------------------------------------------------------
fakedf.to_csv("GettingFakeNewswithMetadata.csv")
