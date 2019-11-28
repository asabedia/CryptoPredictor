import pandas as pd
import numpy as np
def create_date(df):
    new_col = []
    for i in range(len(df)):
        new_col.append(df['time'][i][:df['time'][i].index("T")])
    df['Day'] = new_col
    return df
def split_news(df,coin_type):
    df_new = df[df['Type'] == coin_type][['DatePublished', 'Polarity_of_Desc']]
    return df_new
def merger(df_left,df_right):
    df_merged = pd.merge(df_left, df_right, how = 'left', left_on='Day',right_on='DatePublished').drop(columns = 'DatePublished')
    df_merged.replace('', np.nan, inplace=True)
    df_merged['Polarity_of_Desc'].replace(np.nan,"0", inplace = True)
    return df_merged
df_BTC = create_date(pd.read_csv('csv/historical_data_BTC.csv'))
df_ETH = create_date(pd.read_csv('csv/historical_data_ETH.csv'))
df_LTC = create_date(pd.read_csv('csv/historical_data_LTC.csv'))
df_news = pd.read_csv('csv/news_gb_days.csv')

df_news_BTC = split_news(df_news,'Bitcoin')
df_news_ETH =  split_news(df_news,'Ethereum')
df_news_LTC =  split_news(df_news,'Litecoin')

news_and_BTC = merger(df_BTC,df_news_BTC)
news_and_ETH = merger(df_ETH,df_news_ETH)
news_and_LTC = merger(df_LTC,df_news_LTC)


news_and_BTC.to_csv('csv/BTC_with_news.csv')
news_and_ETH.to_csv('csv/ETH_with_news.csv')
news_and_LTC.to_csv('csv/LTC_with_news.csv')

print(news_and_BTC)