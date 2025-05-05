import numpy as np
from astropy.stats import biweight_location, biweight_scale
import clustering_metrics as cm


def calculate_cluster_centers(data, labels):

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

def calculate_distances(data, og_labels, pred_labels, metric = 'location', og_centers = None, pred_centers = None):

    if og_centers is None:
        centers = calculate_cluster_centers(data, og_labels)
        og_centers = np.array([centers[metric] for cluster in centers.values()])
        
    if pred_centers is None:
        centers = calculate_cluster_centers(data, pred_labels)
        pred_centers = np.array([centers[metric] for cluster in centers.values()])
        
    # Validate dimensions
    if cog_centers.shape[1] != pred_centers.shape[1]:
        raise ValueError("The two center arrays must have the same number of dimensions.")

    # Compute the distance matrix
    distance_matrix = np.linalg.norm(og_centers[:, np.newaxis, :] - pred_centers[np.newaxis, :, :], axis=2)

    return distance_matrix

def calculate_areas(data, labels, std = "scale", radiuses = None)
        
    if radiues is None:
        aux = calculate_cluter_centers(data, lables)
        raiduses = np.array([aux[std] for cluster in aux.values()])
        
    areas = pi * (raiduses ** 2)
    
    return areas   
    
def calculate_overlaps(data, og_labels, pred_labels, distances = None, og_radiuses = None, pred_radiuses = None):

    og_centers = None
    pred_centers = None
    metric = "location"
    std = "scale"

    if og_radiuses is None:
        centers = calculate_cluster_centers(data, og_labels)
        og_centers = np.array([centers[metric] for cluster in centers.values()])
        og_raiduses = np.array([centers[std] for cluster in centers.values()])
        
    if pred_radiuses is None:
        centers = calculate_cluster_centers(data, pred_labels)
        pred_centers = np.array([centers[metric] for cluster in centers.values()])
        pred_raiduses = np.array([centers[std] for cluster in centers.values()])
        
    if distances is None:
        
        if og_centers is None:
            centers = calculate_cluster_centers(data, og_labels)
            og_centers = np.array([centers[metric] for cluster in centers.values()])
            

        if pred_centers is None:
            centers = calculate_cluster_centers(data, pred_labels)
            pred_centers = np.array([centers[metric] for cluster in centers.values()])
        
        distances = calculate_distances(data, og_labels, pred_labels, metric = metric, og_centers = og_centers, pred_centers = pred_centers)
        
    num_true, num_pred = distances.shape
    overlap_areas = np.zeros((num_true, num_pred))

    for i in range(num_true):
        for j in range(num_pred):
            R1, R2 = og_radiuses[i], pred_radiuses[j]
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
    overlap_f1 = cm.ger_f1_score(overlap_completeness, overlap_purity)
    
    return overlap_completeness, overlap_purity, overlap_f1

def calculate_membership_metrics(og_labels, pred_labels):
    cm = pd.crosstab(index=og_labels, columns=pred_labels) 
    
    og_groups, og_group_count = np.unique(og_labels, return_counts=True)
    pred_groups, pred_group_count = np.unique(pred_labels, return_counts=True)
    
    ogc_expanded = og_group_count[:, np.newaxis]
    pgc_expanded = pred_group_Count[np.newaxis,:]
    
    completeness = cm/ogc_expanded
    purity = cm/pgc_expanded
    f1_score = cm.get_f1_score(completeness, purity)
    
    return completeness, purity, f1_score

def reliability_gradient(matrixes):
    
    x_values = np.linspace(0, 1, 100)
    
    for x in x_values:
        percentages = []

        for m in matrixes:
            if df.empty:    
                continue
            count_above_x = (m > x).any(axis=0).sum()
            percentage = (count_above_x / m.shape[1]) * 100
            percentages.append(percentage)
        
        # Compute the average percentage across all matrices
        avg_percentage = np.mean(percentages) if percentages else 0
    
    return avg_percentage
