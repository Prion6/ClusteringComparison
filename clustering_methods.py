
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from scipy.cluster.hierarchy import linkage
import hdbscan
import ds
import milaDS
import clustering_external_tools as tools

def run_DSP_test(X_data, Y_data, V_data, Z_clus):

    galaxy_info, grouping, summary = milaDS.DSp_groups(X_data, Y_data, V_data, Z_clus)

    columns = ['idx_gal', 'shID', 'Ngali', 'Ngalf', 'Rij(kpc)', 'size(kpc)', 'sigma(km/s)', 'Pmin(ds)', 'GrNr', 'x(kpc)', 'y(kpc)', 'V_LOS(km/s)']

    grouping_df = pd.DataFrame(grouping, columns=columns)

    V_disp = grouping_df['sigma(km/s)'].to_numpy()

    data = np.vstack((X_data, Y_data, V_data, V_disp)).T

    labels = grouping_df['GrNr'].to_numpy()


    return labels, data, None

def run_DS_test(X_data, Y_data, V_data):

    deltas = ds.DS_test(X_data, Y_data, V_data)

    #print(deltas)

    return deltas

def run_GMM(data, params):
    
    
    clusterer = BayesianGaussianMixture()

    if 'max_clusters' in params:
        clusterer.n_components = params['max_clusters']

    if 'covariance_type' in params:    
        clusterer.covariace_type = params['covariance_type']

    if 'random_state' in params:    
        clusterer.random_state = params['random_state']

    clusterer.fit(data)

    # Get probabilities and labels
    probabilities = clusterer.predict_proba(data)
    labels = probabilities.argmax(axis=1)  # Get labels directly from probabilities

    return labels, probabilities, clusterer

def run_HDBSCAN(data, params): 

    clusterer = hdbscan.HDBSCAN()

    if 'min_cluster_size' in params:
        clusterer.min_cluster_size = params['min_cluster_size']

    labels = clusterer.fit_predict(data)
    # Get probabilities
    probabilities = clusterer.probabilities_


    return labels, probabilities, clusterer

def run_DBSCAN(data, params):
    
    clusterer = DBSCAN()

    key_point = 'location'

    if 'min_cluster_size' in params:
        clusterer.min_samples = params['min_cluster_size']

    if 'key_point' in params:
        key_point = params['key_point']

    # Get sorted distances for the 5th nearest neighbor
    sorted_distances = tools.find_eps(data, clusterer.min_samples)

    elbow_value, elbow_point, location, location_point, median, median_point, mean, mean_point = tools.get_key_points(sorted_distances)

    if key_point == 'location':
        clusterer.eps =  location
    elif key_point == 'elbow_value':
        clusterer.eps =  elbow_value
    elif key_point == 'median':
        clusterer.eps =  median
    elif key_point == 'mean':
        clusterer.eps =  mean

    labels = clusterer.fit_predict(data)

    # Approximate probabilities: 1 / (distance to nearest cluster center + 1e-9)
    #core_samples_mask = np.zeros_like(labels, dtype=bool)
    #core_samples_mask[clusterer.core_sample_indices_] = True
    #core_points = data[core_samples_mask]

    #distances = pairwise_distances(data, core_points, metric='euclidean')
    #probabilities = 1 / (distances.min(axis=1) + 1e-9)

    # Normalize probabilities to sum to 1
    #probabilities = probabilities / probabilities.sum()

    probabilities = np.ones_like(labels, dtype=float)

    return labels, probabilities, clusterer

def run_OPTICS(data, params):


    clusterer = OPTICS()

    
    if 'min_cluster_size' in params:
        clusterer.min_samples = params['min_cluster_size']

    
    if 'max_eps' in params:
        clusterer.max_eps = params['max_eps']
    
    labels = clusterer.fit_predict(data)
 
    # Approximate probabilities from reachability distance
    #reachability = clusterer.reachability_
    #probabilities = np.exp(-reachability)  # Convert reachability to probabilities
    probabilities = np.ones_like(labels, dtype=float)

    return labels, probabilities, clusterer

def run_Kmeans(data, params):
    
    max_clusters = 30
    key_point = 'location'

    clusterer = KMeans()

    if 'random_state' in params:
        clusterer.random_state = params['random_state']

    if 'max_clusters' in params:
        max_clusters = params['max_clusters']

    sse = []
    for k in range(1, max_clusters):
        clusterer.n_clusters = k
        clusterer.fit(data)
        sse.append(clusterer.inertia_)  # SSE

    elbow_value, elbow_point, location, location_point, median, median_point, mean, mean_point = tools.get_key_points(sse)
    
    if key_point == 'location':
        clusterer.n_clusters =  location_point
    elif key_point == 'elbow_value':
        clusterer.n_clusters =  elbow_point
    elif key_point == 'median':
        clusterer.n_clusters =  median_point
    elif key_point == 'mean':
        clusterer.n_clusters =  mean_point

    # Fit the model
    clusterer.fit(data)

    # Predict cluster labels
    labels = clusterer.predict(data)

    # Cluster centers
    centroids = clusterer.cluster_centers_

    # Compute probabilities
    #distances = clusterer.transform(data)  # Distances to each cluster center
    #probabilities = np.exp(-distances)  # Convert distances to "probabilities"
    #probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # Normalize

    probabilities = np.ones_like(labels, dtype=float)

    return labels, centroids, clusterer

def run_Aglomerative_Clustering(data, params):

    # Perform Agglomerative Clustering
    clusterer = AgglomerativeClustering()

    max_clusters = 30
    key_point = 'location'


    if 'linkage' in params:
        clusterer.linkage = params['linkage']

    if 'max_clusters' in params:
        max_clusters = params['max_clusters']

    sse = []
    for k in range(1, max_clusters):
        clusterer.n_clusters = k
        clusterer.fit(data)
        labels = clusterer.labels_
    
        # Compute WCSS (sum of squared distances to cluster centroid)
        cluster_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        wcss_k = sum(np.linalg.norm(data[labels == i] - center, axis=1).sum() for i, center in enumerate(cluster_centers))
        sse.append(wcss_k)

    elbow_value, elbow_point, location, location_point, median, median_point, mean, mean_point = tools.get_key_points(sse)
    
    if key_point == 'location':
        clusterer.n_clusters =  location_point
    elif key_point == 'elbow_value':
        clusterer.n_clusters =  elbow_point
    elif key_point == 'median':
        clusterer.n_clusters =  median_point
    elif key_point == 'mean':
        clusterer.n_clusters =  mean_point

    clusterer.fit(data)
    labels = clusterer.labels_
    linked = linkage(data, params['linkage'])

    #distances = Z[:, 2]  # Distance column in linkage matrix
    #probabilities = 1 / (distances + 1e-9)

    probabilities = np.ones_like(labels, dtype=float)

    return labels, probabilities, clusterer

def run_Affinity_Propagation(data, params):
    # Affinity Propagation model
    clusterer = AffinityPropagation()


    if 'random_state' in params:
        clusterer.random_state = params['random_state']

    clusterer.fit(data)


    # Extract cluster centers and labels
    clusters = clusterer.cluster_centers_
    labels = clusterer.labels_

    # Approximate probabilities from affinity scores
    #affinity = clusterer.affinity_matrix_
    #probabilities = np.exp(affinity)  # Convert affinity to probabilities
    #probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # Normalize

    
    probabilities = np.ones_like(labels, dtype=float)


    return labels, probabilities, clusterer

def run_BIRCH(data, params):
    
    # BIRCH clustering model
    clusterer = Birch()


    if 'n_clusters' in params:
        clusterer.n_clusters = params['n_clusters']

    
    if 'clustering_threshold' in params:
        clusterer.threshold = params['clustering_threshold']

    clusterer.fit(data)

    # Predict cluster labels
    labels = clusterer.predict(data)

    # Approximate probabilities using distances to centroids
    #centroids = clusterer.subcluster_centers_
    #distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    #probabilities = np.exp(-distances)  # Convert distances to "probabilities"
    #probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # Normalize

    
    probabilities = np.ones_like(labels, dtype=float)


    return labels, probabilities, clusterer

