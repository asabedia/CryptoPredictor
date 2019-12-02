from sklearn.cluster import AffinityPropagation, KMeans
import numpy as np
import iso8601 as iso
import pandas as pd
import seaborn as sns; sns.set()
from yellowbrick.cluster import KElbowVisualizer
from datetime import datetime
import matplotlib.pyplot as plt

df_stripped = pd.read_csv('csv/processed/BTC_with_news_daily.csv')
df_stripped['delta'] = df_stripped['close']-df_stripped['open']
df_train = df_stripped.reset_index().drop(columns=['time','Day','index','open','close','low','high','z_score','Polarity_of_Desc','delta'])

def elbow_point(df):
    visualizer = KElbowVisualizer(
        KMeans(), k=(2, 10), metric='calinski_harabasz', timings=False
    )
    visualizer.fit(df)
    visualizer.show()
    plt.close()
    return visualizer.elbow_value_


kmeans = KMeans(n_clusters=elbow_point(df_train)).fit(df_train)
plt.scatter(df_train['price_mean'], df_train['price_stdev'], c=kmeans.predict(df_train), s=50, cmap='viridis')
plt.xlabel('mean')
plt.ylabel('stdev')
plt.show()

# df_stripped['cluster']=kmeans.labels_
# df_stripped.to_csv('cluster_news_delta.csv')
# df_output= df_stripped.reset_index().drop(columns=['index','z_score','price_mean','open','close'])

