import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import clustering_external_tools as ctools
from itertools import combinations
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix


def get_lax_match(true, pred):

    # True Positives: Estimated matches the real value and is positive
    tp = np.sum((true >= 0) & (pred >= 0))
    # False Positives: Estimated is positive but does not match real
    fp = np.sum((true < 0) & (pred >= 0))
    # True Negatives: Both real and estimated are non-positive and match
    tn = np.sum((true < 0) & (pred < 0))
    # False Negatives: Real is positive, but estimated does not match or is non-positive
    fn = np.sum((true >= 0) & (pred < 0))

    return tp, fp, tn, fn

def get_completitud(tp, fn):
    return (tp/(tp + fn))

def get_purity(tp, fp):#recall
    return (tp/(tp + fp))

def get_f1_score(completitud, purity):
    
    return 2 * (completitud * purity) / (completitud + purity)

def get_rand_index(tp, fp, tn, fn):
     return (tp + tn)/(tp + fp + tn + fn)

def get_dunn_index(data, labels):
    """
    Calculate the Dunn Index for clustering.

    Parameters:
    - data (ndarray): 2D array where each row is a data point and each column is a feature.
    - labels (ndarray): 1D array with cluster labels for each data point.

    Returns:
    - dunn_index (float): Computed Dunn Index.
    """
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        raise ValueError("Dunn Index requires at least two clusters.")

    # Calculate intra-cluster diameters (max pairwise distances within each cluster)
    intra_cluster_diameters = []
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            distances = cdist(cluster_points, cluster_points)
            diameter = np.max(distances)
        else:
            diameter = 0  # A single point has zero diameter
        intra_cluster_diameters.append(diameter)

    # Calculate inter-cluster distances (min pairwise distances between clusters)
    inter_cluster_distances = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i >= j:  # Avoid duplicate pairs and self-pairs
                continue
            points_i = data[labels == label_i]
            points_j = data[labels == label_j]
            distances = cdist(points_i, points_j)
            inter_cluster_distances.append(np.min(distances))

    # Dunn Index: ratio of min inter-cluster distance to max intra-cluster diameter
    min_inter_cluster_distance = np.min(inter_cluster_distances)
    max_intra_cluster_diameter = np.max(intra_cluster_diameters)

    if max_intra_cluster_diameter == 0:
        return np.inf  # Avoid division by zero, consider infinite Dunn Index
    
    return min_inter_cluster_distance / max_intra_cluster_diameter

def get_calinski_harabasz_index(data, labels):
    """
    Compute the Calinski-Harabasz Index for clustering.

    Parameters:
    - data (ndarray): 2D array where each row is a data point and each column is a feature.
    - labels (array-like): Cluster labels for each data point.

    Returns:
    - float: Calinski-Harabasz Index.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = data.shape[0]
    
    # Overall mean
    overall_mean = np.mean(data, axis=0)

    # Initialize between-cluster and within-cluster dispersions
    between_dispersion = 0
    within_dispersion = 0

    for label in unique_labels:
        cluster_points = data[labels == label]
        cluster_size = cluster_points.shape[0]
        cluster_mean = np.mean(cluster_points, axis=0)
        
        # Between-cluster dispersion
        between_dispersion += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)
        
        # Within-cluster dispersion
        within_dispersion += np.sum((cluster_points - cluster_mean) ** 2)
    
    # Calculate Calinski-Harabasz Index
    numerator = between_dispersion / (n_clusters - 1)
    denominator = within_dispersion / (n_samples - n_clusters)
    return numerator / denominator






def purity_score(true, predicted):
    # Confusion matrix (contingency matrix)
    cont_matrix = contingency_matrix(true, predicted)
    
    return np.sum(np.max(cont_matrix, axis=0)) / np.sum(cont_matrix)




