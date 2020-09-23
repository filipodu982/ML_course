import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Reading dataset and creating independent and dependent variable matrices
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
#plt.show()

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

y_pred = hc.fit_predict(X)

# Printing clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='green', label='Cluster 3')
#plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')
#plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=100, c='magenta', label='Cluster 5')
#plt.scatter(hc.cluster_centers_[:, 0], hc.cluster_centers_[:, 1], s=300, c='yellow', label='centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score 1-100')
plt.legend()
plt.show()

