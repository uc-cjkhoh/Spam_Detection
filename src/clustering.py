import pandas as pd
import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from src.decorators import error_log, timer


@error_log
@timer
def get_clustering(vectors, cluster_limit=20):
    def get_optimal_n_cluster(data: pd.Series):
        kmeans = KMeans(init='k-means++')
        visualizer = KElbowVisualizer(kmeans, k=(1, cluster_limit))    
        visualizer.fit(data)
        
        return visualizer.elbow_value_
    
    optimal_n_cluster = get_optimal_n_cluster(vectors)
    kmeans = KMeans(n_clusters=optimal_n_cluster)
    cluster_label = kmeans.fit_predict(vectors)
    
    return cluster_label


def get_centroids(vectors: pd.Series):
    pass


def align_clusters(old_centroids, new_centroids):
    """
    Match new clusters to old clusters based on centroid similarity
    """
    # Compute similarity matrix
    similarity = cosine_similarity(old_centroids, new_centroids)
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(-similarity)
    
    # Create mapping: new_label -> old_label
    label_mapping = {new_idx: old_idx for old_idx, new_idx in zip(row_ind, col_ind)}
    
    return label_mapping

# Usage across iterations
iteration_1_centroids = kmeans_1.cluster_centers_
iteration_2_centroids = kmeans_2.cluster_centers_

# Get mapping
mapping = align_clusters(iteration_1_centroids, iteration_2_centroids)

# Remap new labels to match old numbering
new_labels_aligned = np.array([mapping[label] for label in new_labels])