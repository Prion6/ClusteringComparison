import numpy as np
from astropy.stats import biweight_location, biweight_scale
import clustering_metrics as cm
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from numpy import pi
import pandas as pd

def find_eps(data, k=3):
  
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

def calculate_cluster_centers(data, labels):

    # Dictionary to store cluster centers
    cluster_centers = {}

    # Calculate the mean position for each cluster
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]  # Select points belonging to the current cluster
        cluster_centers[label] = {
            'mean': np.mean(cluster_points, axis=0),
            'std': 0.0 if cluster_points.size <= 1 else np.std(cluster_points, axis=0),
            'location': cluster_points[0] if cluster_points.size <= 1 else biweight_location(cluster_points, axis=0),
            'scale': 0.0 if cluster_points.size <= 1 else biweight_scale(cluster_points, axis=0)
        }

    return cluster_centers

def calculate_distances(data, og_labels, pred_labels, metric = 'location', og_centers = None, pred_centers = None):

    if og_centers is None:
        centers = calculate_cluster_centers(data, og_labels)
        og_centers = np.array([centers[cluster][metric] for cluster in centers])
        
        
    if pred_centers is None:
        centers = calculate_cluster_centers(data, pred_labels)
        pred_centers = np.array([centers[cluster][metric] for cluster in centers])
        
    # Validate dimensions
    if og_centers.shape[1] != pred_centers.shape[1]:
        raise ValueError("The two center arrays must have the same number of dimensions.")

    # Compute the distance matrix
    distance_matrix = np.linalg.norm(og_centers[:, np.newaxis, :] - pred_centers[np.newaxis, :, :], axis=2)

    return distance_matrix

def calculate_areas(data, labels, metric = "location", std = "scale", radiuses = None):
        
    if radiuses is None:
        aux = calculate_cluster_centers(data, labels)
        centers = np.array([aux[cluster][metric] for cluster in aux])
        stds = np.array([aux[cluster][std] for cluster in aux])
        radiuses = np.linalg.norm(stds, axis = 1)
        
    areas = pi * (radiuses ** 2)
    
    return areas   
    
def calculate_overlaps(data, og_labels, pred_labels, distances = None, og_radiuses = None, pred_radiuses = None):

    og_centers = None
    pred_centers = None
    metric = "location"
    std = "scale"

    if og_radiuses is None:
        centers = calculate_cluster_centers(data, og_labels)
        og_centers = np.array([centers[cluster][metric] for cluster in centers])
        stds = np.array([centers[cluster][std] for cluster in centers])
        og_radiuses = np.linalg.norm(stds, axis = 1)
        
    if pred_radiuses is None:
        centers = calculate_cluster_centers(data, pred_labels)
        pred_centers = np.array([centers[cluster][metric] for cluster in centers])
        stds = np.array([centers[cluster][std] for cluster in centers])
        pred_radiuses = np.linalg.norm(stds, axis = 1)
        
    if distances is None:
        
        if og_centers is None:
            centers = calculate_cluster_centers(data, og_labels)
            og_centers = np.array([centers[cluster][metric] for cluster in centers])
            

        if pred_centers is None:
            centers = calculate_cluster_centers(data, pred_labels)
            pred_centers = np.array([centers[cluster][metric] for cluster in centers])
        
        distances = calculate_distances(data, og_labels, pred_labels, metric = metric, og_centers = og_centers, pred_centers = pred_centers)
        
    num_true, num_pred = distances.shape
    overlap_areas = np.zeros((num_true, num_pred))

    for i in range(num_true):
        for j in range(num_pred):
            R1 = og_radiuses[i]
            R2 = pred_radiuses[j]
            d = distances[i, j]

            # If the circles do not overlap
            if d >= R1 + R2:
                overlap_areas[i, j] = 0
            # If one circle is completely inside another
            elif d <= abs(R1 - R2):
                overlap_areas[i, j] = pi * min(R1, R2) ** 2
            # Partial overlap case
            else:
                part1 = R1**2 * np.arccos((d**2 + R1**2 - R2**2) / (2 * d * R1))
                part2 = R2**2 * np.arccos((d**2 + R2**2 - R1**2) / (2 * d * R2))
                part3 = 0.5 * np.sqrt((-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2))
                overlap_areas[i, j] = part1 + part2 - part3
    
    return overlap_areas
        
def calculate_overlap_metrics(data, og_labels, pred_labels, og_areas = None, pred_areas = None, overlap_areas = None):
    
    if og_areas is None:
        og_areas = calculate_areas(data, og_labels)
        
    if pred_areas is None:
        pred_areas = calculate_areas(data, pred_labels)
        
    if overlap_areas is None:
        overlap_areas = calculate_overlaps(data, og_labels, pred_labels)
        
    og_areas_expanded = og_areas[:, np.newaxis]  # Shape (num_true, 1)
    pred_areas_expanded = pred_areas[np.newaxis, :]  # Shape (1, num_pred)
        
    overlap_completeness = overlap_areas/og_areas_expanded
    overlap_purity = overlap_areas/pred_areas_expanded
    overlap_f1 = cm.get_f1_score(overlap_completeness, overlap_purity)
    
    overlap_f1 = np.nan_to_num(overlap_f1, nan=0)
    
    return overlap_completeness, overlap_purity, overlap_f1

def calculate_overlap_completeness(data, og_labels, pred_labels, og_areas = None, pred_areas = None, overlap_areas = None):
    
    if og_areas is None:
        og_areas = calculate_areas(data, og_labels)
        
    if pred_areas is None:
        pred_areas = calculate_areas(data, pred_labels)
        
    if overlap_areas is None:
        overlap_areas = calculate_overlaps(data, og_labels, pred_labels)
        
    og_areas_expanded = og_areas[:, np.newaxis]  # Shape (num_true, 1)
        
    overlap_completeness = overlap_areas/og_areas_expanded
    
    return overlap_completeness

def calculate_overlap_purity(data, og_labels, pred_labels, og_areas = None, pred_areas = None, overlap_areas = None):
    
    if og_areas is None:
        og_areas = calculate_areas(data, og_labels)
        
    if pred_areas is None:
        pred_areas = calculate_areas(data, pred_labels)
        
    if overlap_areas is None:
        overlap_areas = calculate_overlaps(data, og_labels, pred_labels)
        
    pred_areas_expanded = pred_areas[np.newaxis, :]  # Shape (1, num_pred)
        
    overlap_purity = overlap_areas/pred_areas_expanded
    
    return overlap_purity

def calculate_overlap_f1(data, og_labels, pred_labels, og_areas = None, pred_areas = None, overlap_areas = None):
    if og_areas is None:
        og_areas = calculate_areas(data, og_labels)
        
    if pred_areas is None:
        pred_areas = calculate_areas(data, pred_labels)
        
    if overlap_areas is None:
        overlap_areas = calculate_overlaps(data, og_labels, pred_labels)
        
    og_areas_expanded = og_areas[:, np.newaxis]  # Shape (num_true, 1)
    pred_areas_expanded = pred_areas[np.newaxis, :]  # Shape (1, num_pred)
        
    overlap_completeness = overlap_areas/og_areas_expanded
    overlap_purity = overlap_areas/pred_areas_expanded
    overlap_f1 = cm.get_f1_score(overlap_completeness, overlap_purity)
    
    overlap_f1 = np.nan_to_num(overlap_f1, nan=0)
    
    return overlap_f1

def calculate_membership_metrics(og_labels, pred_labels):
    confusion_matrix = pd.crosstab(index=og_labels, columns=pred_labels) 
    
    og_groups, og_group_count = np.unique(og_labels, return_counts=True)
    pred_groups, pred_group_count = np.unique(pred_labels, return_counts=True)
    
    ogc_expanded = og_group_count[:, np.newaxis]
    pgc_expanded = pred_group_count[np.newaxis,:]
    
    completeness = confusion_matrix/ogc_expanded
    purity = confusion_matrix/pgc_expanded
    f1_score = cm.get_f1_score(completeness, purity)
    f1_score = np.nan_to_num(f1_score, nan=0)
    
    return completeness, purity, f1_score

def calculate_membership_completeness(og_labels, pred_labels):
    confusion_matrix = pd.crosstab(index=og_labels, columns=pred_labels) 
    
    og_groups, og_group_count = np.unique(og_labels, return_counts=True)
    
    ogc_expanded = og_group_count[:, np.newaxis]
    
    completeness = confusion_matrix/ogc_expanded
    
    return completeness

def calculate_membership_purity(og_labels, pred_labels):
    confusion_matrix = pd.crosstab(index=og_labels, columns=pred_labels) 
    
    pred_groups, pred_group_count = np.unique(pred_labels, return_counts=True)
    
    pgc_expanded = pred_group_count[np.newaxis,:]
    
    purity = confusion_matrix/pgc_expanded
    
    return purity

def calculate_membership_f1(og_labels, pred_labels):
    confusion_matrix = pd.crosstab(index=og_labels, columns=pred_labels) 
    
    og_groups, og_group_count = np.unique(og_labels, return_counts=True)
    pred_groups, pred_group_count = np.unique(pred_labels, return_counts=True)
    
    ogc_expanded = og_group_count[:, np.newaxis]
    pgc_expanded = pred_group_count[np.newaxis,:]
    
    completeness = confusion_matrix/ogc_expanded
    purity = confusion_matrix/pgc_expanded
    f1_score = cm.get_f1_score(completeness, purity)
    f1_score = np.nan_to_num(f1_score, nan=0)
    
    return f1_score

def reliability_gradient(matrix, steps=100, source_axis=1):
    
    x_values = np.linspace(0, 1, steps)
    percentages = []
    
    ev_axis = 0
    if source_axis == 0:
        ev_axis = 1

    for x in x_values:

        count_above_x = (matrix >= x).any(axis=source_axis).sum()
        percentage = (count_above_x / matrix.shape[ev_axis]) * steps
        percentages.append(percentage)
    
    return percentages

def retag_noise(clusters, n = 3, tag = -1, key = "haloId"):
    
    new_cluster_list = []
    for df in clusters:
        counts = df.groupby(key)[key].transform("count")
        df = df.copy()
        df.loc[counts <= n, key] = tag
        new_cluster_list.append(df.reset_index(drop=True))
    return new_cluster_list
