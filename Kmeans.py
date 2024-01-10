import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""============================== Find the good number of cluster with silhouette score and elbow method =============================="""
# Charger les données depuis le fichier data.npy
data = np.load('data.npy')

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Essayer différents nombres de clusters
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialiser les listes pour stocker les résultats des deux méthodes
inertia_values = []  # Pour la méthode du coude
silhouette_scores = []  # Pour le silhouette score

# Calculer l'inertie et le silhouette score pour chaque nombre de clusters
for n_clusters in range_n_clusters:
    # Appliquer le k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)

    # Calculer l'inertie (méthode du coude)
    inertia_values.append(kmeans.inertia_)

    # Calculer le silhouette score
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Trouver le nombre optimal de clusters avec la méthode du coude
optimal_clusters_elbow = range_n_clusters[np.argmin(np.gradient(inertia_values))]

# Trouver le nombre optimal de clusters avec le meilleur silhouette score
optimal_clusters_silhouette = range_n_clusters[np.argmax(silhouette_scores)]

# Moyenne des résultats obtenus par chaque méthode
average_optimal_clusters = (optimal_clusters_elbow + optimal_clusters_silhouette) // 2

# Afficher les résultats
print(f"Nombre optimal de clusters (méthode du coude) : {optimal_clusters_elbow}")
print(f"Nombre optimal de clusters (silhouette score) : {optimal_clusters_silhouette}")
print(f"Moyenne des résultats obtenus par chaque méthode : {average_optimal_clusters}")

# Tracer la courbe du coude (elbow) et le silhouette score en fonction du nombre de clusters
plt.figure(figsize=(12, 4))

# Méthode du coude
plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, inertia_values, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Elbow method')

# Silhouette score
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
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