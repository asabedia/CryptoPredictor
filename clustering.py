from sklearn.cluster import AffinityPropagation
import numpy as np 
import pandas as pd

df = pd.read_csv('csv/processed/processed_cleaned_BTC.csv')
df_stripped= df[449:1165]
df_train = df_stripped.reset_index().drop(columns=['time','index'])
cluster = AffinityPropagation().fit(df_train)
df_clusters = pd.DataFrame(columns = df_train.columns)
for i in cluster.cluster_centers_indices_:
    df_clusters = df_clusters.append(df_train[i])

print(df_clusters)