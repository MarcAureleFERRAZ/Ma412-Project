import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""============================== Finding best parameters Spectral clustering =============================="""
# Charger les données depuis le fichier data.npy
data = np.load('data.npy')

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Essayer différents nombres de clusters
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialiser les listes pour stocker les résultats des deux méthodes
silhouette_scores = []  # Pour le silhouette score

# Calculer le silhouette score pour chaque nombre de clusters
for n_clusters in range_n_clusters:
    # Appliquer le spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
    cluster_labels = spectral.fit_predict(data_scaled)

    # Calculer le silhouette score
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Trouver le nombre optimal de clusters avec le meilleur silhouette score
optimal_clusters_silhouette = range_n_clusters[np.argmax(silhouette_scores)]

# Afficher les résultats
print(f"Nombre optimal de clusters (silhouette score) : {optimal_clusters_silhouette}")

# Tracer le silhouette score en fonction du nombre de clusters
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Spectral Clustering')
plt.show()



"""============================== Spectral clustering =============================="""
# Load the aircraft trajectory data
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Choose the number of clusters based on your problem
n_clusters = 3  # You may need to tune this parameter

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')  # You can choose an appropriate affinity metric
labels = spectral.fit_predict(data_scaled)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the results
# (You may need to choose appropriate dimensions for visualization based on your data)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=10, cmap='viridis')
plt.title('Aircraft Trajectory Clustering with Spectral Clustering')
plt.show()
