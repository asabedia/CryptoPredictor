import math
import requests
from textblob import TextBlob
import pandas as pd

df = pd.DataFrame(columns=['Type', 'Title', 'DatePublished'])

url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/NewsSearchAPI"
headers = {
    'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
    'x-rapidapi-key': "YOURAPIKEY"
    }
############################### First Pass to Get Count of WebPages ########################################
querystring1 = {"autoCorrect":"false","pageNumber":"1","pageSize":"50","q":"Bitcoin","safeSearch":"false",
               "fromPublishedDate":"2019-01-01"}
querystring2 = {"autoCorrect":"false","pageNumber":"1","pageSize":"50","q":"Ethereum","safeSearch":"false",
               "fromPublishedDate":"2019-01-01"}
querystring3 = {"autoCorrect":"false","pageNumber":"1","pageSize":"50","q":"Litecoin","safeSearch":"false",
                "fromPublishedDate":"2019-01-01"}
############################### First Pass to Get Count of WebPages ########################################


response_Bitcoin = requests.request("GET", url, headers=headers, params=querystring1)
response_Ethereum = requests.request("GET", url, headers=headers, params=querystring2)
response_Litecoin = requests.request("GET", url, headers=headers, params=querystring3)


page_numbers_BTC = math.ceil(response_Bitcoin.json()['totalCount']/50)
page_numbers_ETH = math.ceil(response_Ethereum.json()['totalCount']/50)
page_numbers_LTC = math.ceil(response_Litecoin.json()['totalCount']/50)


for i in range(1,page_numbers_BTC):
    querystring1 = {"autoCorrect": "false", "pageNumber": i, "pageSize": "50", "q": "Bitcoin", "safeSearch": "false",
                    "fromPublishedDate": "2019-01-01"}
    response_Bitcoin = requests.request("GET", url, headers=headers, params=querystring1)
    json = response_Bitcoin.json()['value']
    for i in range(len(json)):
        df = df.append({'Type': "Bitcoin", 'Title': json[i]['title'], 'Description': json[i]['description'],
                        'Body': json[i]['body']
                           , 'DatePublished': json[i]['datePublished'][:json[i]['datePublished'].index("T")],
                        'Polarity_of_Title': TextBlob(json[i]['title']).sentiment.polarity,
                        'Polarity_of_Desc': TextBlob(json[i]['description']).sentiment.polarity,
                        'Polarity_of_Body': TextBlob(json[i]['body']).sentiment.polarity}, ignore_index=True)

for i in range(1,page_numbers_ETH):
    querystring2 = {"autoCorrect": "false", "pageNumber": i, "pageSize": "50", "q": "Ethereum", "safeSearch": "false",
                    "fromPublishedDate": "2019-01-01"}
    response_Ethereum = requests.request("GET", url, headers=headers, params=querystring2)
    json = response_Ethereum.json()['value']
    for i in range(len(json)):
        df = df.append({'Type': "Ethereum", 'Title': json[i]['title'], 'Description': json[i]['description'],
                        'Body': json[i]['body']
                           , 'DatePublished': json[i]['datePublished'][:json[i]['datePublished'].index("T")],
                        'Polarity_of_Title': TextBlob(json[i]['title']).sentiment.polarity,
                        'Polarity_of_Desc': TextBlob(json[i]['description']).sentiment.polarity,
                        'Polarity_of_Body': TextBlob(json[i]['body']).sentiment.polarity}, ignore_index=True)
for i in range(1,page_numbers_LTC):
    querystring3 = {"autoCorrect": "false", "pageNumber": i, "pageSize": "50", "q": "Litecoin", "safeSearch": "false",
                    "fromPublishedDate": "2019-01-01"}
    response_Litecoin = requests.request("GET", url, headers=headers, params=querystring3)
    json = response_Litecoin.json()['value']
    for i in range(len(json)):
        df = df.append({'Type': "Litecoin", 'Title': json[i]['title'], 'Description': json[i]['description'],
                        'Body': json[i]['body']
                           , 'DatePublished': json[i]['datePublished'][:json[i]['datePublished'].index("T")],
                        'Polarity_of_Title': TextBlob(json[i]['title']).sentiment.polarity,
                        'Polarity_of_Desc': TextBlob(json[i]['description']).sentiment.polarity,
                        'Polarity_of_Body': TextBlob(json[i]['body']).sentiment.polarity}, ignore_index=True)

df.to_csv('csv/news.csv',index=False)


corr_body_desc = df['Polarity_of_Body'].corr(df['Polarity_of_Desc'])
corr_body_title = df['Polarity_of_Body'].corr(df['Polarity_of_Title'])
corr_desc_title = df['Polarity_of_Desc'].corr(df['Polarity_of_Title'])

print(f"Body to Description Correlation: {corr_body_desc}")
print(f"Body to Title Correlation: {corr_body_title}")
print(f"Title to Description Correlation: {corr_desc_title}")

gb = df.groupby(['Type','DatePublished']).mean()
gb.to_csv('csv/news_gb_days.csv')
