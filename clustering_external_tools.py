import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from astropy.stats import biweight_location, biweight_scale
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def compute_mapping_probabilities(array1, array2):
    """
    Compute a probability matrix for mapping categories between two arrays.
    """
    # Get unique categories
    categories_1 = np.unique(array1)
    categories_2 = np.unique(array2)
    
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((len(categories_1), len(categories_2)))
    
    # Map values to indices
    map_1 = {val: idx for idx, val in enumerate(categories_1)}
    map_2 = {val: idx for idx, val in enumerate(categories_2)}
    
    # Count occurrences of (category1, category2) pairs
    for val1, val2 in zip(array1, array2):
        confusion_matrix[map_1[val1], map_2[val2]] += 1
    
    # Normalize the confusion matrix to get probabilities
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    probability_matrix = confusion_matrix / row_sums
    
    return probability_matrix, map_1, map_2

def map_values(array1, array2, direction="array1_to_array2"):
    """
    Map values between two arrays based on their category probabilities.
    
    Parameters:
    - array1, array2: Input arrays with categories to map.
    - direction: "array1_to_array2" or "array2_to_array1" for the desired mapping.
    
    Returns:
    - Transformed array based on the mapping.
    - The mapping dictionary.
    """
    # Compute the probability matrix
    prob_matrix, map_1, map_2 = compute_mapping_probabilities(array1, array2)
    
    if direction == "array1_to_array2":
        source_map = map_1
        target_map = map_2
        prob_source_to_target = prob_matrix
    elif direction == "array2_to_array1":
        source_map = map_2
        target_map = map_1
        prob_source_to_target = prob_matrix.T
    else:
        raise ValueError("Invalid direction. Use 'array1_to_array2' or 'array2_to_array1'.")
    
    # Reverse maps for lookup
    reverse_source_map = {idx: val for val, idx in source_map.items()}
    reverse_target_map = {idx: val for val, idx in target_map.items()}
    
    # Determine the mapping
    mapping = {}
    for source_idx in range(prob_source_to_target.shape[0]):
        target_idx = np.argmax(prob_source_to_target[source_idx])  # Most likely target
        mapping[reverse_source_map[source_idx]] = reverse_target_map[target_idx]
    
    # Transform the array based on the mapping
    if direction == "array1_to_array2":
        transformed_array = np.array([mapping[val] for val in array1])
    else:
        transformed_array = np.array([mapping[val] for val in array2])
    
    return transformed_array, mapping


def find_eps(data, k=3):
    """
    Find a suitable value for `eps` using the k-nearest neighbors algorithm.
    
    Parameters:
    - X: Data array of shape (n_samples, n_features).
    - k: Number of neighbors to consider.

    Returns:
    - distances: Sorted distances to the k-th nearest neighbor.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    return np.sort(distances[:, k-1])

def get_key_points(data):
    # Combine X_data, Y_data, Z_data into a single array

    knee_locator = KneeLocator(range(len(data)), data, curve="convex", direction="increasing")

    # Get the elbow point
    elbow_point = knee_locator.knee
    elbow_value = data[elbow_point]

    location = biweight_location(data)
    median = np.median(data)
    mean = np.mean(data)

    # Find the indices of location, median, and mean points
    location_point = np.argmin(np.abs(data - location))
    median_point = np.argmin(np.abs(data - median))
    mean_point = np.argmin(np.abs(data - mean))

    return elbow_value, elbow_point, location, location_point, median, median_point, mean, mean_point


def map_labels_with_mismatch(true_labels, predicted_labels):
    # Compute the contingency matrix
    contingency = confusion_matrix(true_labels, predicted_labels)
    num_true, num_pred = contingency.shape
    
    # Pad the contingency matrix with zeros to make it square
    padded_contingency = np.zeros((max(num_true, num_pred), max(num_true, num_pred)))
    padded_contingency[:num_true, :num_pred] = contingency
    
    # Use Hungarian algorithm to find the best alignment
    row_ind, col_ind = linear_sum_assignment(-padded_contingency)  # Maximize match by minimizing -contingency
    
    # Create a mapping from predicted labels to true labels
    mapping = {col: row for row, col in zip(row_ind, col_ind) if col < num_pred and row < num_true}
    
    # Map predicted labels to their aligned true labels
    aligned_predicted_labels = np.array([mapping[label] if label in mapping else -1 for label in predicted_labels])
    return aligned_predicted_labels, mapping, contingency


def get_cluster_center(data, labels):

    # Dictionary to store cluster centers
    cluster_centers = {}

    # Calculate the mean position for each cluster
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]  # Select points belonging to the current cluster
        cluster_centers[label] = {
            'mean': np.mean(cluster_points, axis=0),
            'std': np.std(cluster_points, axis=0),
            'location': biweight_location(cluster_points, axis=0),
            'scale': biweight_scale(cluster_points, axis=0)
        }

    return cluster_centers


def compute_distance_matrix(centers1, centers2):

    # Validate dimensions
    if centers1.shape[1] != centers2.shape[1]:
        raise ValueError("The two center arrays must have the same number of dimensions.")

    # Compute the distance matrix
    distance_matrix = np.linalg.norm(centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :], axis=2)

    return distance_matrix

def matrix_as_table(matrix, true_prefix="True", pred_prefix="Pred", decimals=4):
    """
    Prints a matrix as a formatted table with indexed rows and columns.

    Parameters:
        matrix (np.ndarray): The 2D NumPy array to be formatted.
        true_prefix (str): Prefix for row labels (true groups).
        pred_prefix (str): Prefix for column labels (predicted groups).
        decimals (int): Number of decimal places to round to.
    """
    num_true, num_pred = matrix.shape
    df = pd.DataFrame(matrix.round(decimals), 
                      columns=[f"{pred_prefix} {j}" for j in range(num_pred)], 
                      index=[f"{true_prefix} {i}" for i in range(num_true)])
    
    return df

def get_distance_matrix(data, true_labels, pred_labels, metric = 'location'):
    true_centers = get_cluster_center(data, true_labels)
    pred_centers = get_cluster_center(data, pred_labels)

    true_mean_centers = np.array([cluster[metric] for cluster in true_centers.values()])
    pred_mean_centers = np.array([cluster[metric] for cluster in pred_centers.values()])

    distances = compute_distance_matrix(true_mean_centers, pred_mean_centers)

    return distances

def get_overlapping_areas(distances, radius1, radius2):
    num_true, num_pred = distances.shape
    overlap_areas = np.zeros((num_true, num_pred))

    for i in range(num_true):
        for j in range(num_pred):
            R1, R2 = radius1[i], radius2[j]
            d = distances[i, j]

            # If the circles do not overlap
            if d >= R1 + R2:
                overlap_areas[i, j] = 0
            # If one circle is completely inside another
            elif d <= abs(R1 - R2):
                overlap_areas[i, j] = np.pi * min(R1, R2) ** 2
            # Partial overlap case
            else:
                part1 = R1**2 * np.arccos((d**2 + R1**2 - R2**2) / (2 * d * R1))
                part2 = R2**2 * np.arccos((d**2 + R2**2 - R1**2) / (2 * d * R2))
                part3 = 0.5 * np.sqrt((-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2))
                overlap_areas[i, j] = part1 + part2 - part3