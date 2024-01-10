# Import necessary libraries
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""
==============================
Finding best parameters Gaussian Mixture models (GMM)
==============================
"""

# Load data from the file 'data.npy'
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Try different number of components (clusters)
range_n_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialize lists to store the results of the two methods
silhouette_scores = []  # for the silhouette score

# Calculate the silhouette score for each number of components
for n_components in range_n_components:
    # Apply the Gaussian Mixture model
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data_scaled)

    # Calculate the silhouette score
    cluster_labels = gmm.predict(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the optimal number of components with the best silhouette score
optimal_components_silhouette = range_n_components[np.argmax(silhouette_scores)]

# Display the results
print(f"Optimal number of components (silhouette score) : {optimal_components_silhouette}")

# Plot the silhouette score as a function of the number of components
plt.figure(figsize=(12, 4))

# Silhouette score
plt.plot(range_n_components, silhouette_scores, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

"""
==============================
Gaussian Mixture model (GMM)
==============================
"""

# Load data from the file 'data.npy'
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Choose the number of components based on your problem
n_components = 6

# Apply the Gaussian Mixture model (GMM)
gmm = GaussianMixture(n_components=n_components, random_state=42)
labels = gmm.fit_predict(data_scaled)

# Evaluate the clustering performance using the silhouette score
silhouette_avg = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=10, cmap='viridis')
plt.title('Aircraft Trajectory Clustering (GMM)')
plt.show()