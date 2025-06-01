"""
Title: Gap Statistic for Optimal Clusters
Author: Kavinder Roghit Kanthen
Date: 03-03-2025
Last Modified: 03-03-2025
Description:
This script uses the **Gap Statistic Method** to determine the optimal number 
of clusters (K) by comparing actual clustering with random reference datasets. 
It calculates the Gap value and selects the K that maximizes it, offering a 
statistically sound alternative to the Elbow and Silhouette methods.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

def compute_gap_statistic(X, k_max=10, B=10):
    """
    Computes the Gap Statistic for clustering
    """
    def within_cluster_dispersion(X, labels, n_clusters):
        """Computes the sum of squared distances for given cluster labels."""
        dispersion = 0
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 1:
                dispersion += np.sum(pairwise_distances(cluster_points).mean())
        return dispersion
    
    gap_values = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        actual_dispersion = within_cluster_dispersion(X, kmeans.labels_, k)
        # Generate B random datasets
        reference_disps = []
        for _ in range(B):
            X_random = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            kmeans_random = KMeans(n_clusters=k, random_state=42).fit(X_random)
            reference_dispersion = within_cluster_dispersion(X_random, kmeans_random.labels_, k)
            reference_disps.append(reference_dispersion)

        gap = np.log(np.mean(reference_disps)) - np.log(actual_dispersion)
        gap_values.append(gap)
    
    optimal_k = np.argmax(gap_values) + 1
    
    return optimal_k, gap_values

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=42)

# Compute the Gap Statistic
optimal_k, gap_values = compute_gap_statistic(X, k_max=10)

# Plot the Gap Statistic result
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), gap_values, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Gap Statistic")
plt.title("Gap Statistic Method for Finding Optimal Clusters")
plt.axvline(x=optimal_k, linestyle="--", color="red", label=f"Optimal K = {optimal_k}")
plt.legend()
plt.show()