import json
import pandas as pd
from textblob import TextBlob

## PURPOSE OF TESTING ##
f1=open("BTC.txt", "r")
BTC =f1.read()
BTC = BTC.replace("<b>", "")
BTC = BTC.replace("</b>", "")
f1.close()

f2=open("ETH.txt", "r")
ETH =f2.read()
ETH = ETH.replace("<b>", "")
ETH = ETH.replace("</b>", "")
f2.close()

f3=open("LTC.txt", "r")
LTC =f3.read()
LTC = LTC.replace("<b>", "")
LTC = LTC.replace("</b>", "")
f3.close()


btc_json= json.loads(BTC)['value']
eth_json= json.loads(ETH)['value']
ltc_json= json.loads(LTC)['value']

## PURPOSE OF TESTING ##

df = pd.DataFrame(columns=['Type','Title','DatePublished'])

for i in range(len(btc_json)):
    df = df.append({'Type':'Bitcoin','Title':btc_json[i]['title'],'Description':btc_json[i]['description'],'Body':btc_json[i]['body']
                   ,'DatePublished':btc_json[i]['datePublished'][:btc_json[i]['datePublished'].index("T")],'Polarity_of_Title':TextBlob(btc_json[i]['title']).sentiment.polarity,
                    'Polarity_of_Desc':TextBlob(btc_json[i]['description']).sentiment.polarity,
                    'Polarity_of_Body':TextBlob(btc_json[i]['body']).sentiment.polarity},ignore_index=True)

for i in range(len(eth_json)):
    df = df.append({'Type': 'Ethereum', 'Title': eth_json[i]['title'], 'Description': eth_json[i]['description'],
                    'Body': eth_json[i]['body']
                       , 'DatePublished': eth_json[i]['datePublished'][:eth_json[i]['datePublished'].index("T")],
                    'Polarity_of_Title': TextBlob(eth_json[i]['title']).sentiment.polarity,
                    'Polarity_of_Desc': TextBlob(eth_json[i]['description']).sentiment.polarity,
                    'Polarity_of_Body': TextBlob(eth_json[i]['body']).sentiment.polarity}, ignore_index=True)

for i in range(len(ltc_json)):
    df = df.append({'Type': 'Litecoin', 'Title': ltc_json[i]['title'], 'Description': ltc_json[i]['description'],
                    'Body': ltc_json[i]['body']
                       , 'DatePublished': ltc_json[i]['datePublished'][:ltc_json[i]['datePublished'].index("T")],
                    'Polarity_of_Title': TextBlob(ltc_json[i]['title']).sentiment.polarity,
                    'Polarity_of_Desc': TextBlob(ltc_json[i]['description']).sentiment.polarity,
                    'Polarity_of_Body': TextBlob(ltc_json[i]['body']).sentiment.polarity}, ignore_index=True)

df.to_csv('csv/news.csv',index=False)


corr_body_desc = df['Polarity_of_Body'].corr(df['Polarity_of_Desc'])
corr_body_title = df['Polarity_of_Body'].corr(df['Polarity_of_Title'])
corr_desc_title = df['Polarity_of_Desc'].corr(df['Polarity_of_Title'])

print(f"Body to Description Correlation: {corr_body_desc}")
print(f"Body to Title Correlation: {corr_body_title}")
print(f"Title to Description Correlation: {corr_desc_title}")

gb = df.groupby(['Type','DatePublished']).mean()
gb.to_csv('csv/news_gb_days.csv')
