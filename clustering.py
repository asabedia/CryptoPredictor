from sklearn.cluster import AffinityPropagation, KMeans
import numpy as np
import iso8601 as iso
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from datetime import datetime

df = pd.read_csv('csv/processed/processed_cleaned_BTC.csv')
df_stripped = df[449:1165]
df_train = df_stripped.reset_index().drop(columns=['time','index','z_score','price_mean','open','close'])

def affinity():
    cluster = AffinityPropagation().fit(df_train)
    df_clusters = pd.DataFrame(columns = df_train.columns)
    df_clusters['time'] =  0
    for i in cluster.cluster_centers_indices_:
        df_clusters = df_clusters.append(df_train.iloc[i])
        df_clusters.loc[i,'time'] = df_stripped.iloc[i]['time']
    return df_clusters
def elbow_point(df):
    visualizer = KElbowVisualizer(
        KMeans(), k=(2, 10), metric='calinski_harabasz', timings=False
    )
    visualizer.fit(df)
    return visualizer.elbow_value_

def km():

    kmeans = KMeans(n_clusters=elbow_point(df_train)).fit(df_train)
    print(kmeans.cluster_centers_)
    return kmeans.labels_

df_output= df_stripped.reset_index().drop(columns=['index','z_score','price_mean','open','close'])
df_output['Cluster_Assigned'] = km()
df_output['Hour'] = df_output['time'].apply(lambda x: iso.parse_date(x).hour)

gb = df_output.groupby(['Cluster_Assigned','Hour']).size()
# print(gb)
gb.to_csv('csv/group_by_hours.csv')