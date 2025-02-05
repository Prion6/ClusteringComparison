from sklearn.neighbors import NearestNeighbors
import numpy as np
from astropy.stats import biweight_location
from astropy.stats import biweight_scale

def DS_test(X_data, Y_data, V_data, num_neighbors = 10):


    # Compute global mean velocity and velocity dispersion
    mean_velocity = biweight_location(V_data)
    velocity_dispersion = biweight_scale(V_data)

    # Example: positions of galaxies (2D for simplicity: could be RA, Dec)
    positions = np.column_stack((X_data, Y_data))  # replace with your position data

    # Number of neighbors to consider
    k = num_neighbors + 1

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(positions)
    distances, indices = nbrs.kneighbors(positions)

    local_means = []
    local_stds = []

    for i in range(len(positions)):
        # Extract the velocities of the nearest neighbors (excluding itself)
        neighbor_velocities = V_data[indices[i][1:]]
        local_means.append(biweight_location(neighbor_velocities))
        local_stds.append(biweight_scale(neighbor_velocities))

    local_means = np.array(local_means)
    local_stds = np.array(local_stds)

    deltas = (k+1) * ((local_means - mean_velocity)**2 + (local_stds - velocity_dispersion)**2) / velocity_dispersion

    return deltas

def monte_carlo_simulation(X_data, Y_data, V_data, num_neighbors = 10, num_simulations=1000):
    simulated_deltas = []
    
    for _ in range(num_simulations):
        shuffled_velocities = np.random.permutation(V_data)
        deltas = DS_test(X_data, Y_data, shuffled_velocities, num_neighbors)
        simulated_deltas.append(deltas)
        
    return np.array(simulated_deltas)


def find_outliers_by_percentage(delta_values, delta_simulations, significance_level=0.05):
    """
    Identify outliers based on top percentage of DS values.
    delta_values: array of DS values for galaxies
    delta_simulations: 2D array of DS values from Monte Carlo simulations
    significance_level: top percentage threshold (e.g., 0.05 for top 5%)
    Returns indices of galaxies considered outliers.
    """
    # Calculate the threshold based on significance level
    flat_deltas = delta_simulations.flatten()  # Flatten all simulated values
    threshold = np.percentile(flat_deltas, 100 * (1 - significance_level))
    
    # Find outliers in actual delta values
    outlier_indices = np.where(delta_values > threshold)[0]
    
    return outlier_indices, threshold


def find_outliers_by_sigma(deltas, montecarlo_results):
    # Extract Monte Carlo biweight stats
    montecarlo_locations = np.array([biweight_location(mc) for mc in montecarlo_results])
    montecarlo_scales = np.array([biweight_scale(mc) for mc in montecarlo_results])
    
    # Estimate thresholds using Monte Carlo
    mean_location = np.mean(montecarlo_locations)
    mean_scale = np.mean(montecarlo_scales)
    scale_uncertainty = np.std(montecarlo_scales)

    # Define thresholds using Monte Carlo results
    threshold_upper = mean_location + 3 * mean_scale + scale_uncertainty
    threshold_lower = mean_location - 3 * mean_scale - scale_uncertainty

    # Identify outliers
    outliers = np.where((deltas > threshold_upper) | (deltas < threshold_lower))[0]

    return outliers, threshold_upper, threshold_lower

#estudiar estimadores de biweight, articulo 1990
