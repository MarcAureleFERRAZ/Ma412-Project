import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

"""============================== Find the good number of cluster with silhouette score =============================="""
# Charger les données depuis le fichier data.npy
data = np.load('data.npy')
# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Essayer différents nombres de clusters et calculer le silhouette score
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
silhouette_scores = []

for n_clusters in range_n_clusters:
    # Appliquer le clustering hiérarchique
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(data_scaled)

    # Calculer le silhouette score
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Trouver le nombre optimal de clusters avec le meilleur silhouette score
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]

# Afficher le résultat
print(f"Le nombre optimal de clusters est : {optimal_n_clusters}")

# Tracer la courbe du silhouette score en fonction du nombre de clusters
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score en fonction du nombre de clusters')
plt.show()


"""============================== Hierarchical Clustering =============================="""
# Standardize the data (important for clustering)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


# Choose the number of clusters based on your problem
n_clusters = 6

# Apply hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
labels = clustering.fit_predict(data_scaled)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=10, cmap='viridis')
plt.title('Aircraft Trajectory Clustering')
plt.show()

# Display the dendrogram
linked = linkage(data_scaled, 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.show()

print(data.shape)