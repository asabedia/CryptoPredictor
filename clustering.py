from sklearn.cluster import AffinityPropagation, KMeans
import tkinter
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from yellowbrick.cluster import KElbowVisualizer
from datetime import datetime
import matplotlib.pyplot as plt

df_stripped = pd.read_csv('csv/processed/BTC_with_news_daily.csv')
df_stripped['delta'] = df_stripped['close']-df_stripped['open']
df_train = df_stripped.reset_index().drop(columns=['time','Day','index','open','close','low','high','z_score'])

def elbow_point(df):
    visualizer = KElbowVisualizer(
        KMeans(), k=(2, 10), metric='calinski_harabasz', timings=False
    )
    visualizer.fit(df)
    visualizer.show()
    plt.close()
    return visualizer.elbow_value_


kmeans = KMeans(random_state=3425,n_clusters=elbow_point(df_train)).fit(df_train[['price_mean','price_stdev']])
df_train['labels']=kmeans.labels_
df_train['color']=kmeans.labels_

for i in range(len(df_train)):
    if(df_train['color'].iloc[i]==0):
        df_train['color'].loc[i]='red'
    if(df_train['color'].iloc[i]==1):
        df_train['color'].loc[i] = 'green'
    if (df_train['color'].iloc[i] == 2):
        df_train['color'].loc[i] = 'blue'
    if (df_train['color'].iloc[i] == 3):
        df_train['color'].loc[i] = 'yellow'
    if (df_train['color'].iloc[i] == 4):
        df_train['color'].loc[i] = 'purple'

## SCATTER PLOT CLUSTER POLARITY/DELTA ##
# plt.scatter(df_train['Polarity_of_Desc'], df_train['delta'], c=df_train['color'], s=50, edgecolor='k')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.xlabel('Polarity_of_Desc')
# plt.ylabel('Delta')
# plt.title('Polarity_of_Desc/Delta Clusters')
# plt.show()
# plt.close()

## SCATTER PLOT CLUSTER STDEV/MEAN ##
plt.scatter(df_train['price_mean'], df_train['price_stdev'], c=df_train['color'], s=50, edgecolor='k')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('price_mean')
plt.ylabel('price_stdev')
plt.title('Mean/Stdev Clusters')
plt.show()
plt.close()

## SCATTER PLOT TIME CLUSTER STDEV/MEAN ##
df_time = df_stripped.iloc[::-1]
df_time['cluster']=df_train['labels'].iloc[::-1]
df_time['color']=df_train['color'].iloc[::-1]
plt.scatter(df_time['Day'], df_time['color'], c=df_time['color'], s=50,edgecolor='k')
plt.xlabel('time')
plt.ylabel('cluster')
plt.xticks(df_time['Day'][::14], rotation='vertical')
plt.title('Mean/Stdev Clusters by Time')
plt.show()
plt.close()


# SCATTER PLOT WITH TREND RISK STDEV/MEAN##
df_train['risk per dollar']=  df_train['price_stdev']/df_train['price_mean']
plt.scatter(df_train['price_mean'], df_train['risk per dollar'], c=df_train['color'], s=50,edgecolor='k')
z = np.polyfit(df_train['price_mean'], df_train['risk per dollar'], 1)
p = np.poly1d(z)
plt.plot(df_train['price_mean'],p(df_train['price_mean']),c='black')
plt.xlabel('price_mean')
plt.ylabel('risk per dollar')
plt.title('Mean vs Risk Per Dollar')
plt.show()
plt.close()