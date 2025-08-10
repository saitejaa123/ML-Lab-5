import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("features.csv") 

# Select features (ignore target column)
X_clust = df[['f1', 'f2']]

# Store silhouette scores
k_values = range(2, 10)  # Try k from 2 to 9
sil_scores = []

# Loop through different k values
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_clust)
    score = silhouette_score(X_clust, kmeans.labels_)
    sil_scores.append(score)
    print(f"k = {k}, Silhouette Score = {score:.4f}")

# Plot the silhouette scores
plt.figure(figsize=(8,5))
plt.plot(k_values, sil_scores, marker='o', color='blue')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k')
plt.grid(True)
plt.show()
