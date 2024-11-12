"""Elbow Visualiser"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#take new line in jupyter
data = pd.read_csv('sales_data_sample.csv', sep = ',', encoding = 'Latin-1')
#take new line in jupyter
data
#take new line in jupyter
selected_features = data[['QUANTITYORDERED','PRICEEACH']]
selected_features
#take new line in jupyter
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_features = scaler.fit_transform(selected_features)
#take new line in jupyter
 normalized_features
#take new line in jupyter
wcss = [] 
for i in range(1, 11):
kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10,random_state=0)
kmeans.fit(normalized_features)
wcss.append(kmeans.inertia_)
#take new line in jupyter
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#end
