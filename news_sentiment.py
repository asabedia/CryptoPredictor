<<<<<<< HEAD
from newsapi import NewsApiClient
import pandas as pd
import math
newsapi = NewsApiClient(api_key='f01b6cd465054722ab0cdaa55ce5f8ef')


i=0
page_size = 100
all_articles = newsapi.get_everything(q='bitcoin',
                                      from_param='2019-09-29',
                                      to='2019-10-28',
                                      language='en',
                                      sort_by='relevancy',
                                      page = 1,
                                      page_size = page_size)
total_results = all_articles['totalResults']
total_pages = math.ceil(total_results/page_size)
print(total_pages)
articles = [article['title'] for article in all_articles['articles']]
print(articles)


for i in range (2,total_pages+1):
    continue


=======
from newsapi import newsApiClient
>>>>>>> creating a wrapper for coinbase api
