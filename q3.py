import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("features.csv")  

# Select only features (ignore target column)
X_clust = df[['f1', 'f2']]

# Perform K-Means clustering (k=2)
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X_clust)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(X_clust.iloc[:,0], X_clust.iloc[:,1],
            c=kmeans.labels_, cmap='viridis', alpha=0.6, edgecolors='k')

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=300, c='red', marker='X', label='Centroids')

plt.xlabel('f1')
plt.ylabel('f2')
plt.title('K-Means Clustering (k=2)')
plt.legend()
plt.show()

# Print cluster info
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Cluster Labels:\n", kmeans.labels_)
