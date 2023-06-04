from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

credit_data = pd.read_csv('../dataset/original/extracted.csv')
y = credit_data['Churn_Label']
X = credit_data.drop(columns='Churn_Label')
X.drop(columns='CLIENTNUM', inplace=True)

kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
cluster_label = pd.DataFrame(kmeans.labels_, columns=['Cluster_Label'])
print(cluster_label.value_counts())
X = pd.concat([X, y, cluster_label], axis=1)

clusters = []
for k in range(6):
    customer_cluster = X.loc[X['Cluster_Label'] == k]
    clusters.append(customer_cluster)
for k in range(6):
    print(clusters[k].shape)
# for k in range(6):
#     cluster = clusters[k].drop(columns='Cluster_Label', inplace=False)
#     cluster.to_csv(f'../dataset/before_smote/cluster_{k+1}.csv', index=False)
