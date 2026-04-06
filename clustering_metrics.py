import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from itertools import combinations
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
import pandas as pd
import my_data_manager as mdm

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

def compute_times(reported_results):
    rows = []

    for method, alg in reported_results.items():
        df_time = alg["time"].copy()

        # Remove iteration column if present
        if "Iteration" in df_time.columns:
            df_time = df_time.drop(columns=["Iteration"])

        # Convert all cells to numeric
        df_time = df_time.apply(pd.to_numeric, errors="coerce")

        # Sum all galaxy-cluster times within each execution/run
        # Each column corresponds to one run
        run_totals = df_time.sum(axis=0, skipna=True).values

        # Remove invalid totals if any
        run_totals = run_totals[~np.isnan(run_totals)]

        mean_time = np.mean(run_totals) if len(run_totals) > 0 else np.nan
        std_time = np.std(run_totals) if len(run_totals) > 0 else np.nan

        rows.append({
            "Method": method,
            "Mean": mean_time,
            "Std": std_time,
            "Summary": f"{mean_time:.4f} ± {std_time:.4f}" if len(run_totals) > 0 else "nan ± nan"
        })

    return pd.DataFrame(rows)

def compute_performance_metric(reported_results, samples, metric_func):
    results_dict = {}

    for method, alg_report in reported_results.items():
        #print(method)
        executions = alg_report["executions"]
        clustering_results = mdm.transpose_list_of_dfs(executions)

        #print(clustering_results.keys())

        cluster_metrics = {}

        for galaxy_cluster in samples:
            s = str(galaxy_cluster["firstHaloInFOFGroupId"].iloc[0])
            results = clustering_results[s]

            metric_clus = []

            for exec in results.columns:
                labels = results[exec].dropna().to_numpy(dtype=np.float64)
                true_labels = galaxy_cluster["haloId"].to_numpy(dtype=np.float64)

                assert len(labels) == len(true_labels), (
                    f"Error in cluster {s} for method {method} due to length mismatch: "
                    f"{len(labels)} vs {len(true_labels)}"
                )

                metric_clus.append(metric_func(true_labels, labels))

            cluster_metrics[s] = metric_clus

        df_method = pd.DataFrame(dict(cluster_metrics))
        results_dict[method] = df_method

    return results_dict

def compute_clustering_metric(reported_results, samples, metric_func):

    results_dict = {}

    for method, alg_report in reported_results.items():
        executions = alg_report["executions"]
        clustering_results = mdm.transpose_list_of_dfs(executions)

        cluster_metrics = {}

        for galaxy_cluster in samples:
            s = str(galaxy_cluster["firstHaloInFOFGroupId"].iloc[0])
            results = clustering_results[s]

            metric_clus = []

            for exec in results.columns:
                labels = results[exec].dropna().to_numpy(dtype=np.float64)
                data = galaxy_cluster[["RA", "DEC"]].to_numpy(dtype=np.float64)

                assert len(labels) == len(data), (
                    f"Error in cluster {s} for method {method} due to length mismatch: "
                    f"{len(labels)} vs {len(data)}"
                )

                try:
                    score = metric_func(data, labels)
                    metric_clus.append(score)
                except ValueError:
                    metric_clus.append(np.nan)

            cluster_metrics[s] = metric_clus

        df_method = pd.DataFrame(dict(cluster_metrics))
        results_dict[method] = df_method

    return results_dict

def compute_spatial_metric(reported_results, samples, metric_func):
    
    results_dict = {}

    for method, alg_report in reported_results.items():
        executions = alg_report["executions"]
        clustering_results = mdm.transpose_list_of_dfs(executions)

        cluster_metrics = {}

        for galaxy_cluster in samples:
            s = str(galaxy_cluster["firstHaloInFOFGroupId"].iloc[0])
            results = clustering_results[s]

            metric_clus = []

            for exec in results.columns:

                labels = results[exec].dropna().to_numpy(dtype=np.float64)
                true_labels = galaxy_cluster["haloId"].to_numpy(dtype=np.float64)

                data = galaxy_cluster[["RA", "DEC"]].to_numpy(dtype=np.float64)

                metric = metric_func(data, true_labels, labels)

                metric_clus.append(metric)

            cluster_metrics[s] = metric_clus

        results_dict[method] = cluster_metrics

    return results_dict

def compute_membership_metric(reported_results, samples, metric_func):

    results_dict = {}

    for method, alg_report in reported_results.items():
        executions = alg_report["executions"]
        clustering_results = mdm.transpose_list_of_dfs(executions)

        cluster_metrics = {}

        for galaxy_cluster in samples:

            s = str(galaxy_cluster["firstHaloInFOFGroupId"].iloc[0])
            results = clustering_results[s]

            metric_clus = []

            for exec in results.columns:

                labels = results[exec].dropna().to_numpy(dtype=np.float64)
                true_labels = galaxy_cluster["haloId"].to_numpy(dtype=np.float64)

                metric = metric_func(true_labels, labels)

                metric_clus.append(metric)

            cluster_metrics[s] = metric_clus
        
        results_dict[method] = cluster_metrics
    return results_dict

def calculate_pred_gradient(matrix, mode, has_noise_row=True, is_partitioning=False):
    arr = np.array(matrix)
    rows, cols = arr.shape

    try:
        # Predicted substructures are stored by columns
        actual_clusters = arr if is_partitioning else arr[:, 1:]

        if mode == "all":
            clean = actual_clusters

        elif mode == "no-noise":
            clean = actual_clusters[1:, :] if has_noise_row else actual_clusters

        elif mode == "satellites":
            start_row = 2 if has_noise_row else 1
            clean = (
                actual_clusters[start_row:, :]
                if rows > start_row
                else np.empty((0, actual_clusters.shape[1]))
            )

        elif mode == "center":
            center_idx = 1 if has_noise_row else 0
            clean = (
                actual_clusters[center_idx:center_idx+1, :]
                if rows > center_idx
                else np.empty((0, actual_clusters.shape[1]))
            )

        else:
            return [0.0] * 100

        if clean.size == 0 or clean.shape[1] == 0:
            return [0.0] * 100

        thresholds = np.linspace(0, 1, 100)
        gradients = []

        for t in thresholds:
            # Success per predicted substructure (column)
            column_success = np.any(clean >= t, axis=0)
            gradients.append(column_success.mean())

        return gradients

    except Exception:
        return [0.0] * 100








