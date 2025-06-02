import color_generator as cg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#TODO let them choose color pallete
def cluster_plot_2D(X_data, Y_data, labels,
                            noise_label = -1,
                            title="Cluster Plot", cluster_nature="Pred",
                            x_axis_nature="X Axis", y_axis_nature="Y Axis",
                            size=30, scale_array = None, marker="o", colors=None,
                            edges=True, linewidth=1.5, edgecolors=None,
                            figsize=(8,6), grid=True,
                            color_generator=cg.high_contrast_colors):

    #scale_array = None -> no scaling, everything at specified size
    if scale_array is None:
        scale_array = np.ones_like(X_data)

    #careful, clusters should keep their label
    clusters = np.unique(labels, return_inverse=True)[0]

    clusters = clusters[clusters != noise_label]

    if colors is None:
        colors = color_generator(len(clusters))
        
    maped_colors = map_colors(labels, colors, noise_label)
    
    maped_edgecolors = []
    
    if edges:
        if edgecolors is None:
            edgecolors = color_generator(len(clusters))
        maped_edgecolors = map_colors(labels, edgecolors, noise_label)


    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)  # Create figure and axes

    # Scatter plot with custom colors, sizes, and alpha
    scatter = ax.scatter(
        X_data, Y_data, 
        s=scale_array*size, 
        c=maped_colors, 
        edgecolors=maped_edgecolors, 
        linewidth=linewidth,
        marker=marker
    )

    # Create a custom legend mapping group indexes to colors
    handles = []
    
    
    handle = plt.Line2D(
        [0], [0], 
        marker='o', color='w', 
        markerfacecolor='gray', 
        markersize=10, label=f'{cluster_nature} -1'
    )
    
    handles.append(handle)
    
    for cluster in clusters:
        c = int(cluster)
        handle = plt.Line2D(
            [0], [0], 
            marker='o', color='w', 
            markerfacecolor=colors[c], 
            markersize=10, label=f'{cluster_nature} {c}'
        )
        handles.append(handle)
    
    # Add the custom legend
    ax.legend(handles=handles, title='Group Index', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add labels and title
    ax.set_xlabel(x_axis_nature)
    ax.set_ylabel(y_axis_nature)
    ax.set_title(title)

    # Show grid
    if grid:
        ax.grid()

    return fig, ax


def map_colors(labels, colors, noise_label):
    maped_colors = []
    
    for label in labels:
        i = int(label)
        if i == noise_label:  
            maped_colors.append('gray')
        else:
            maped_colors.append(colors[i])
            
    return maped_colors

def gradients_plot(legends, gradients, xlabel = "Reliability", ylabel = "Saples Validated", title = "Reliability Gradient", colors = None):
    
    if colors is None:
        colors = cg.high_contrast_colors(len(gradients))

    x_values = np.linspace(0, 1, 100)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each algorithm's data with a unique color
    for i, (alg, gradient) in enumerate(zip(legends, gradients)):
        ax.plot(x_values, gradient, label=alg, color=colors[i % len(colors)])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    ax.legend()

    # Return figure and axis instead of showing the plot
    return fig, ax

def gradients_bar_plot(legends, gradients, xlabel="Reliability", ylabel="Samples Validated", title="Reliability Gradient", colors = None):
    x_points = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    num_reliabilities = len(x_points)
    num_algorithms = len(gradients)

    legends = list(legends)
    
    group_width = 0.8
    bar_width = group_width/num_algorithms
    fig, ax = plt.subplots(figsize=(12, 6))
    if colors is None:
        colors = cg.high_contrast_colors(len(gradients))

    xticks = []
    xtick_labels = []

    # Plot bars
    
    for i, gradient in enumerate(gradients):
        heights = []
        for rel in x_points:
            full_x = np.linspace(0, 1, len(gradient))
            y = np.interp(rel, full_x, gradient)
            heights.append(y)

        # X positions for this reliability level
        x_offsets = np.arange(len(x_points)) - (group_width / 2) + i * bar_width + bar_width / 2
        ax.bar(x_offsets, heights, width=bar_width, color=colors[i], label=f"{rel}")

        xticks.extend(x_offsets)
        xtick_labels.extend([legends[i]] * num_reliabilities)

    # Set tick labels for reliability values (above)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=90)

    # Add algorithm names centered under each group
    for idx, name in enumerate(x_points):
        group_center = idx
        ax.text(group_center, -0.25 * ax.get_ylim()[1], name,
                ha='center', va='top', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_xlabel(xlabel, labelpad = 40)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    #ax.legend(legends, title="Algorithms")
    ax.grid(True, axis='y')

    return fig, ax

