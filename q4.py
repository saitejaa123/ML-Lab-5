import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load dataset
df = pd.read_csv("features.csv")  

# Select only features (ignore target column)
X_clust = df[['f1', 'f2']]

# Perform K-Means clustering (k=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_clust)

# Calculate clustering scores
sil = silhouette_score(X_clust, kmeans.labels_)
ch = calinski_harabasz_score(X_clust, kmeans.labels_)
db = davies_bouldin_score(X_clust, kmeans.labels_)

# Display results
print(f"Silhouette Score: {sil:.4f}")
print(f"Calinski-Harabasz Score: {ch:.4f}")
print(f"Davies-Bouldin Index: {db:.4f}")
