import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
df = pd.read_csv("features.csv")  

# Select features (ignore target column)
X_clust = df[['f1', 'f2']]

# Store distortions (inertia values)
distortions = []
k_range = range(2, 20)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_clust)
    distortions.append(kmeans.inertia_)

# Compute the differences between distortions
diffs = np.diff(distortions)
diff_ratios = np.abs(diffs[1:] / diffs[:-1])  # ratio of consecutive drops

# Find k where drop slows the most (smallest ratio after a large drop)
optimal_k = k_range[np.argmin(diff_ratios) + 1]  # +1 because diff shortens list by 1

print(f"Optimal number of clusters (Elbow Method) = {optimal_k}")

# Plot elbow curve
plt.figure(figsize=(8,5))
plt.plot(k_range, distortions, marker='o')
plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.legend()
plt.grid(True)
plt.show()