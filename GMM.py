import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""============================== Finding best parameters Gaussian Mixture models (GMM) =============================="""
# Charger les données depuis le fichier data.npy
data = np.load('data.npy')

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Essayer différents nombres de composants (clusters)
range_n_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialiser les listes pour stocker les résultats des deux méthodes
silhouette_scores = []  # Pour le silhouette score

# Calculer le silhouette score pour chaque nombre de composants
for n_components in range_n_components:
    # Appliquer le modèle de mélange de gaussiennes
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data_scaled)


    # Calculer le silhouette score
    cluster_labels = gmm.predict(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)


# Trouver le nombre optimal de composants avec le meilleur silhouette score
optimal_components_silhouette = range_n_components[np.argmax(silhouette_scores)]



# Afficher les résultats
print(f"Nombre optimal de composants (silhouette score) : {optimal_components_silhouette}")


# Tracer le silhouette score en fonction du nombre de composants
plt.figure(figsize=(12, 4))


# Silhouette score
plt.plot(range_n_components, silhouette_scores, marker='o')
plt.xlabel('Nombre de composants')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

"""==============================  Gaussian Mixture model (GMM) =============================="""
# Charger les données depuis le fichier data.npy
data = np.load('data.npy')

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Choisir le nombre de composants basé sur votre problème
n_components = 6

# Appliquer le modèle de mélange de gaussiennes (GMM)
gmm = GaussianMixture(n_components=n_components, random_state=42)
labels = gmm.fit_predict(data_scaled)

# Évaluer la performance du clustering en utilisant le silhouette score
silhouette_avg = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualiser les résultats
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=10, cmap='viridis')
plt.title('Aircraft Trajectory Clustering (GMM)')
plt.show()
