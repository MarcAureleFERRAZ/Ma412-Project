# Import necessary libraries
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import davies_bouldin_score

# Load database
data = np.load('data.npy')

# Perform Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=6) # Set number of clusters
agglomerative_labels = agglomerative.fit_predict(data) # Fit and transform data
agglomerative_dbi = davies_bouldin_score(data, agglomerative_labels) # Calculate Davies-Bouldin Index

# Perform KMeans Clustering
kmeans = KMeans(n_clusters=4) # Set number of clusters
kmeans_labels = kmeans.fit_predict(data) # Fit and transform data
kmeans_dbi = davies_bouldin_score(data, kmeans_labels) # Calculate Davies-Bouldin Index

# Perform Spectral Clustering
spectral = SpectralClustering(n_clusters=3) # Set number of clusters
spectral_labels = spectral.fit_predict(data) # Fit and transform data
spectral_dbi = davies_bouldin_score(data, spectral_labels) # Calculate Davies-Bouldin Index

# Perform Bayesian Gaussian Mixture Model
bayesian_gmm = BayesianGaussianMixture(n_components=6) # Set number of components
bayesian_gmm_labels = bayesian_gmm.fit_predict(data) # Fit and transform data
bayesian_gmm_dbi = davies_bouldin_score(data, bayesian_gmm_labels) # Calculate Davies-Bouldin Index

# Perform Gaussian Mixture Model
gmm = GaussianMixture(n_components=6) # Set number of components
gmm_labels = gmm.fit_predict(data) # Fit and transform data
gmm_dbi = davies_bouldin_score(data, gmm_labels) # Calculate Davies-Bouldin Index

# Display the Davies-Bouldin Index for each clustering method
print(f"Agglomerative DBI: {agglomerative_dbi}")
print(f"KMeans DBI: {kmeans_dbi}")
print(f"Spectral DBI: {spectral_dbi}")
print(f"Bayesian GMM DBI: {bayesian_gmm_dbi}")
print(f"GMM DBI: {gmm_dbi}")