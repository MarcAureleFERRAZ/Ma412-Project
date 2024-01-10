import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

"""============================== Find the good number of cluster with BIC =============================="""
# Charger les données depuis le fichier data.npy
data = np.load('data.npy')

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Essayer différents nombres de composants
range_n_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialiser les listes pour stocker les résultats
silhouette_scores = []  # Pour le silhouette score

# Calculer le silhouette score pour chaque nombre de composants
for n_components in range_n_components:
    # Appliquer le Bayesian Gaussian Mixture
    bgm = BayesianGaussianMixture(n_components=n_components, random_state=42)
    bgm.fit(data_scaled)

    # Calculer le silhouette score
    cluster_labels = bgm.predict(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Trouver le nombre optimal de composants avec le meilleur silhouette score
optimal_components_silhouette = range_n_components[np.argmax(silhouette_scores)]

# Afficher les résultats
print(f"Nombre optimal de composants (silhouette score) : {optimal_components_silhouette}")

# Tracer la courbe du silhouette score en fonction du nombre de composants
plt.figure(figsize=(6, 4))

# Silhouette score
plt.plot(range_n_components, silhouette_scores, marker='o')
plt.xlabel('Nombre de composants')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()



"""============================== Bayesian clustering =============================="""
# Load the aircraft trajectory data
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Choose the number of components (clusters) based on your problem
n_components = 6

# Apply Bayesian Gaussian Mixture Model
bgmm = BayesianGaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
labels = bgmm.fit_predict(data_scaled)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the results
# (You may need to choose appropriate dimensions for visualization based on your data)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=10, cmap='viridis')
plt.title('Aircraft Trajectory Clustering with Bayesian Gaussian Mixture Model')
plt.show()
