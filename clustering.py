from sklearn.cluster import AffinityPropagation, KMeans
import numpy as np 
import pandas as pd

df = pd.read_csv('csv/processed/processed_cleaned_BTC.csv')
df_stripped= df[449:1165]
df_train = df_stripped.reset_index().drop(columns=['time','index','z_score','price_mean','open','close'])

def affinity():
    cluster = AffinityPropagation().fit(df_train)
    df_clusters = pd.DataFrame(columns = df_train.columns)
    df_clusters['time'] =  0
    for i in cluster.cluster_centers_indices_:
        df_clusters = df_clusters.append(df_train.iloc[i])
        df_clusters.loc[i,'time'] = df_stripped.iloc[i]['time']
    return df_clusters

def km():
    kmeans = KMeans(n_clusters=10).fit(df_train)
    print(kmeans.cluster_centers_)
km()