import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

"""============================== Find the good number of components with BIC =============="""
# Load data from the file 'data.npy'
data = np.load('data.npy')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Try different number of components
range_n_components = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Initialize lists to store the results
silhouette_scores = []  # for silhouette score

# Calculate the silhouette score for each number of components
for n_components in range_n_components:
   # Apply Bayesian Gaussian Mixture
   bgm = BayesianGaussianMixture(n_components=n_components, random_state=42)
   bgm.fit(data_scaled)  # Fit the model to the data

   # Calculate the silhouette score
   cluster_labels = bgm.predict(data_scaled)
   silhouette_avg = silhouette_score(data_scaled, cluster_labels)
   silhouette_scores.append(silhouette_avg)

# Find the optimal number of components with the best silhouette score
optimal_components_silhouette = range_n_components[np.argmax(silhouette_scores)]

# Display the results
print(f"Optimal number of components (silhouette score) : {optimal_components_silhouette}")

# Plot the silhouette score as a function of the number of components
plt.figure(figsize=(6, 4))

# Silhouette score
plt.plot(range_n_components, silhouette_scores, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

"""============================== Bayesian clustering =============================="""
# Load the aircraft trajectory data
data = np.load('data.npy')  # assuming data is stored in a numpy array file

# Standardize the data to ensure that each feature has zero mean and unit variance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Choose the number of components (clusters) based on your problem
n_components = 6  # you can choose this value based on your knowledge of the data

# Apply Bayesian Gaussian Mixture Model
# We use 'full' covariance type to allow for variable covariance across clusters
# and set a random state for reproducibility
bgmm = BayesianGaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
labels = bgmm.fit_predict(data_scaled)  # fit the model to the data and get cluster labels

# Evaluate clustering performance using silhouette score
# The silhouette score measures the quality of the clustering by comparing the similarity of a point to its own cluster (cohesion)
# and to other clusters (separation)
silhouette_avg = silhouette_score(data_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")  # print the average silhouette score

# Visualize the results
# We choose the first two dimensions of the scaled data for visualization
# You may need to choose appropriate dimensions based on your data
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, marker='o', s=10, cmap='viridis')
plt.title('Aircraft Trajectory Clustering with Bayesian Gaussian Mixture Model')
plt.show()