import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""============================== Find the optimal number of clusters with silhouette score and elbow method =============================="""
# Load data from the file 'data.npy'
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Try different numbers of clusters
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialize lists to store the results of both methods
inertia_values = []  # For the elbow method
silhouette_scores = []  # For the silhouette score

# Calculate inertia and silhouette score for each number of clusters
for n_clusters in range_n_clusters:
   # Apply k-means
   kmeans = KMeans(n_clusters=n_clusters, random_state=42)
   kmeans.fit(data_scaled)

   # Calculate inertia (elbow method)
   inertia_values.append(kmeans.inertia_)

   # Calculate silhouette score
   cluster_labels = kmeans.labels_
   silhouette_avg = silhouette_score(data_scaled, cluster_labels)
   silhouette_scores.append(silhouette_avg)

# Find the optimal number of clusters using the elbow method
optimal_clusters_elbow = range_n_clusters[np.argmin(np.gradient(inertia_values))]

# Find the optimal number of clusters using the best silhouette score
optimal_clusters_silhouette = range_n_clusters[np.argmax(silhouette_scores)]

# Average the results obtained by each method
average_optimal_clusters = (optimal_clusters_elbow + optimal_clusters_silhouette) // 2

# Display the results
print(f"Optimal number of clusters (elbow method) : {optimal_clusters_elbow}")
print(f"Optimal number of clusters (silhouette score) : {optimal_clusters_silhouette}")
print(f"Average of the results obtained by each method : {average_optimal_clusters}")

# Plot the elbow curve and the silhouette score against the number of clusters
plt.figure(figsize=(12, 4))

# Elbow method
plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, inertia_values, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Silhouette score
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

"""============================== K-means =============================="""
# Load the aircraft trajectory data
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Choose the number of clusters based on your problem
n_clusters = 4  # You may need to tune this parameter

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(data_scaled)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the results
# (You may need to choose appropriate dimensions for visualization based on your data)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=10, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('Aircraft Trajectory Clustering with K-Means')
plt.legend()
plt.show()