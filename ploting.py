
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
import clustering_external_tools as ctools

def generate_high_contrast_colors(n=30, min_diff=16, padding=32):
    
    # Define the range of valid RGB values with padding
    valid_range = np.arange(padding, 256 - padding, min_diff)
    
    # Generate all possible combinations of RGB within the valid range
    possible_colors = np.array(
        [[r, g, b] for r in valid_range for g in valid_range for b in valid_range]
    )
    
    # Shuffle the possible colors to randomize selection
    np.random.shuffle(possible_colors)
    
    # Ensure the first color is gray (mid-gray: [128, 128, 128])
    gray_color = np.array([[128, 128, 128]])  # Mid-gray in RGB
    possible_colors = np.vstack([gray_color, possible_colors])  # Add gray as the first color
    
    # Remove duplicate gray from the pool of possible colors
    _, unique_indices = np.unique(possible_colors, axis=0, return_index=True)
    possible_colors = possible_colors[np.sort(unique_indices)]
    
    # Select the first `n` colors and normalize to [0, 1] range for matplotlib
    selected_colors = possible_colors[:n] / 255.0
    
    # Convert to RGBA by adding an alpha channel
    colors = np.hstack([selected_colors, np.ones((selected_colors.shape[0], 1))])
    
    return colors

def plot_with_gradient(X_data, Y_data, V_disp, threshold=0, scale=100, alpha_scale=1, legend = 'Cluster Substructures'):
    # Normalize the dispersion values for color mapping

    V_disp_scaled = V_disp ** alpha_scale

    norm = Normalize(vmin=np.min(V_disp), vmax=np.max(V_disp))

    # Create a colormap
    cmap = cm.coolwarm  # You can choose different colormaps (e.g., plt.cm.plasma)

    # Map the dispersion values to colors
    colors = cmap(norm(V_disp_scaled))

    filtered_X = X_data[V_disp >= threshold]
    filtered_Y = Y_data[V_disp >= threshold]
    filtered_dispersion_values = V_disp[V_disp >= threshold]

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))  # Create figure and axes

    # Scatter plot with empty circles
    scatter = ax.scatter(filtered_X, filtered_Y, s=filtered_dispersion_values*scale, facecolors=colors , edgecolors='none', linewidth=0)

    # Add colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)  # Specify the axes for the colorbar
    cbar.set_label('Dispersion Value')  # Label for the colorbar

    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(legend)

    # Show grid
    ax.grid()

    # Show the plot
    plt.show()

def plot_clusters(X_data, Y_data, labels, alpha_scale=1, scale=100, legend="Dispersion Plot", edgecolors=None, linewidth=1.5, marker="o", size_array = None):

    if size_array is None:
        size_array = np.ones_like(X_data)

    color_indexes = np.unique(labels, return_inverse=True)[1]

    colors = generate_high_contrast_colors(len(color_indexes))
    edge_colors = generate_high_contrast_colors(len(color_indexes))

    # Normalize size values if needed (optional)
    sizes = (scale ** alpha_scale)

    # Map the color indexes to the color array
    selected_colors = colors[color_indexes]

    # Map the color indexes to the edge color array if provided
    if edgecolors is None:
        selected_edge_colors = edge_colors[color_indexes]
    else:
        selected_edge_colors = edgecolors  # Use default edge color if not provided

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))  # Create figure and axes

    # Scatter plot with custom colors, sizes, and alpha
    scatter = ax.scatter(
        X_data, Y_data, 
        s=size_array * sizes, 
        c=selected_colors, 
        edgecolors=selected_edge_colors, 
        linewidth=linewidth,
        marker=marker
    )

    # Create a custom legend mapping group indexes to colors
    unique_groups = np.unique(color_indexes)
    handles = []
    for group in unique_groups:
        handle = plt.Line2D(
            [0], [0], 
            marker='o', color='w', 
            markerfacecolor=colors[group], 
            markersize=10, label=f'Group {group}'
        )
        handles.append(handle)
    
    # Add the custom legend
    ax.legend(handles=handles, title='Group Index', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(legend)

    # Show grid
    ax.grid()

    return fig, ax

def plot_prob_with_data_and_custom_colors(X_data, Y_data, size_array, color_array, color_indexes, labels, prob, alpha_scale=1, scale=100, 
    legend="Dispersion Plot", edgecolors='black', linewidth=1.5, edge_color_array=None, sigma_levels = 3, ellipse_line_width = 3, linestyle = '--'):
    """
    Use `plot_with_custom_colors` to plot data points and overlay Gaussian ellipses for each group.

    Parameters:
        X_data: X coordinates of data points.
        Y_data: Y coordinates of data points.
        probabilities: Probabilities of data points (optional, for coloring).
        labels: Group labels for each data point.
        color_array: Array of colors corresponding to each group.
        scale: Scaling factor for marker sizes.
        alpha_scale: Exponent to scale probabilities for visualization (if provided).
        legend: Title for the plot.
    """
    # Call `plot_with_custom_colors` to plot the scatter points
    fig, ax = plot_clusters(
        X_data=X_data,
        Y_data=Y_data,
        size_array=size_array,
        color_array=color_array,
        color_indexes=color_indexes,
        alpha_scale=alpha_scale,
        scale=scale,
        legend=legend,
        edgecolors=edgecolors,
        linewidth=linewidth,
        edge_color_array=edge_color_array
    )

    
    # Add concentric ellipses for each group
    unique_labels = np.unique(color_indexes)
    for label in unique_labels:
        # Extract points belonging to the current group
        group_mask = color_indexes == label
        X_group = X_data[group_mask]
        Y_group = Y_data[group_mask]

        # Calculate the mean and covariance for the group
        mean_x, mean_y = np.mean(X_group), np.mean(Y_group)
        cov_matrix = np.cov(X_group, Y_group)

        # Eigen decomposition for ellipse orientation and axes lengths
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        width, height = 2 * np.sqrt(eigenvalues)  # 1-sigma widths
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # Plot concentric ellipses
        for sigma in range(1, sigma_levels + 1):
            ellipse = Ellipse(
                (mean_x, mean_y),
                width * sigma,  # Scale width by sigma
                height * sigma,  # Scale height by sigma
                angle=angle,
                edgecolor=color_array[label],
                facecolor='none',
                linestyle=linestyle,
                linewidth=ellipse_line_width,
                alpha=0.7
            )
            ax.add_patch(ellipse)

    # Add a legend for the middle lines and show the plot
    ax.legend()
    return fig, ax

def plot_comparative_clusters(X_data, Y_data, true_labels, pred_labels,  alpha_scale=1, scale=100, 
    legend="Comparative Plot", edgecolors='black', linewidth=1.5, true_markers = 'x', pre_markers = 'o', 
    base = 'true', average = 'location', deviation = 'scale', size_array = None):

    if size_array is None:
        size_array = np.ones_like(X_data)

    data = np.column_stack((X_data, Y_data))

    true_centers = ctools.get_cluster_center(data, true_labels)
    pred_centers = ctools.get_cluster_center(data, pred_labels)

    true_mean_centers = np.array([cluster[average] for cluster in true_centers.values()])
    pred_mean_centers = np.array([cluster[average] for cluster in pred_centers.values()])

    pred_radial_std = np.array([np.linalg.norm(cluster[deviation]) for cluster in pred_centers.values()])
    true_radial_std = np.array([np.linalg.norm(cluster[deviation]) for cluster in true_centers.values()])

    true_x, true_y = true_mean_centers[:, 0], true_mean_centers[:, 1]
    pred_x, pred_y = pred_mean_centers[:, 0], pred_mean_centers[:, 1]


    if base == 'true':
        labels = true_labels
    elif base == 'pred':
        labels = pred_labels

    fig, ax = plot_clusters(X_data, Y_data, labels, scale=100, legend='GMM Groups', linewidth=0, size_array = size_array)

    ax.scatter(true_x, true_y, label='True Centers (x)', color='red', marker='x', s=100)
    ax.scatter(pred_x, pred_y, label='True Centers (0)', color='blue', marker='o', s=20)

    # Add circles for pred_radial_std
    for px, py, std in zip(pred_x, pred_y, pred_radial_std):
        circle = plt.Circle((px, py), std, color='blue', alpha=0.2, fill=True)
        ax.add_artist(circle)

    for tx, ty, std in zip(true_x, true_y, true_radial_std):
        circle = plt.Circle((tx, ty), std, color='red', alpha=0.2, fill=True)
        ax.add_artist(circle)

    # Update legend, grid, and axis properties
    ax.set_title(legend)
    ax.legend()
    ax.grid(True)
    ax.axis('equal')  # Ensure circles look circular

    return fig, ax