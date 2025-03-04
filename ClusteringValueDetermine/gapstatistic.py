import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

def compute_gap_statistic(X, k_max=10, B=10):
    def within_cluster_dispersion(X, labels, n_clusters):
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
