import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import DBSCAN, OPTICS, KMeans, AgglomerativeClustering, AffinityPropagation, Birch
from scipy.cluster.hierarchy import linkage
import hdbscan
import cluster_utils as cu


def run_GMM(data, params):
    clusterer = BayesianGaussianMixture(
        n_components=params.get('max_clusters', 40),
        covariance_type=params.get('covariance_type', 'full'),
        random_state=params.get('random_state', 0),
        max_iter=params.get('max_iter',1000)
    )
    clusterer.fit(data)
    probabilities = clusterer.predict_proba(data)
    labels = probabilities.argmax(axis=1)
    return labels, probabilities, clusterer


def run_HDBSCAN(data, params):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.get('min_cluster_size', 5)
    )
    labels = clusterer.fit_predict(data)
    probabilities = clusterer.probabilities_
    return labels, probabilities, clusterer


def run_DBSCAN(data, params):
    key_point = params.get('key_point', 'location')
    min_samples = params.get('min_cluster_size', 5)

    sorted_distances = cu.find_eps(data, min_samples)
    elbow_value, elbow_point, location, location_point, median, median_point, mean, mean_point = cu.get_key_points(sorted_distances)

    if key_point == 'location':
        eps = location
    elif key_point == 'elbow_value':
        eps = elbow_value
    elif key_point == 'median':
        eps = median
    elif key_point == 'mean':
        eps = mean
    else:
        eps = location

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(data)
    probabilities = np.ones_like(labels, dtype=float)
    return labels, probabilities, clusterer


def run_OPTICS(data, params):
    clusterer = OPTICS(
        min_samples=params.get('min_cluster_size', 5),
        max_eps=params.get('max_eps', np.inf)
    )
    labels = clusterer.fit_predict(data)
    probabilities = np.ones_like(labels, dtype=float)
    return labels, probabilities, clusterer


def run_Kmeans(data, params):
    max_clusters = params.get('max_clusters', 40)
    key_point = params.get('key_point', 'elbow_value')
    random_state = params.get('random_state', 0)

    sse = []
    for k in range(1, max_clusters):
        clusterer = KMeans(n_clusters=k, random_state=random_state)
        clusterer.fit(data)
        sse.append(clusterer.inertia_)

    elbow_value, elbow_point, location, location_point, median, median_point, mean, mean_point = cu.get_key_points(sse)

    n_clusters = elbow_point

    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusterer.fit(data)
    labels = clusterer.predict(data)
    probabilities = np.ones_like(labels, dtype=float)
    return labels, probabilities, clusterer


def run_Aglomerative_Clustering(data, params):
    max_clusters = params.get('max_clusters', 40)
    key_point = params.get('key_point', 'elbow_value')
    linkage_type = params.get('linkage', 'ward')

    sse = []
    for k in range(1, max_clusters):
        clusterer = AgglomerativeClustering(n_clusters=k, linkage=linkage_type)
        clusterer.fit(data)
        labels = clusterer.labels_
        cluster_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        wcss_k = sum(np.linalg.norm(data[labels == i] - center, axis=1).sum() for i, center in enumerate(cluster_centers))
        sse.append(wcss_k)

    elbow_value, elbow_point, location, location_point, median, median_point, mean, mean_point = cu.get_key_points(sse)

    n_clusters = elbow_point

    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    clusterer.fit(data)
    labels = clusterer.labels_
    probabilities = np.ones_like(labels, dtype=float)
    return labels, probabilities, clusterer


def run_Affinity_Propagation(data, params):
    clusterer = AffinityPropagation(
        random_state=params.get('random_state', 0)
    )
    clusterer.fit(data)
    labels = clusterer.labels_
    probabilities = np.ones_like(labels, dtype=float)
    return labels, probabilities, clusterer
