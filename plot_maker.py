import color_generator as cg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

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

def gradient_bar_plot(
    df,
    metric_name,
    thresholds,
    methods_order,
    ax=None,
    colors=None,
    cmap_name="tab10",
    ymax=0.6,
    text_top_margin=0.025,
    text_gap=0.025,
    show_ylabel=False,
    bar_width=0.75
):
    if len(thresholds) == 0:
        raise ValueError("thresholds must contain at least one value")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    thresholds = sorted(thresholds)

    if colors is None:
        cmap = plt.get_cmap(cmap_name, len(thresholds))
        colors = [cmap(i) for i in range(len(thresholds))]

    if len(colors) < len(thresholds):
        raise ValueError(
            f"Not enough colors for {len(thresholds)} thresholds. "
            f"Received {len(colors)} colors."
        )

    methods_present = [m for m in methods_order if m in df.index]

    if len(methods_present) == 0:
        print(f"No valid methods found for {metric_name}")
        return fig, ax, False

    df = df.loc[methods_present]

    ncols = df.shape[1]
    threshold_grid = np.linspace(0, 1, ncols)
    x = np.arange(len(methods_present))

    threshold_values = {}
    for t in thresholds:
        idx = np.abs(threshold_grid - t).argmin()
        threshold_values[t] = df.iloc[:, idx].values / 100.0

    for j, t in enumerate(thresholds):
        ax.bar(
            x,
            threshold_values[t],
            width=bar_width,
            color=colors[j],
            label=fr"$t \geq {t}$",
            edgecolor="white",
            linewidth=0.5
        )

    text_top = ymax - text_top_margin

    for row_idx, t in enumerate(thresholds):
        y_text = text_top - row_idx * text_gap

        for xi, val in zip(x, threshold_values[t]):
            ax.text(
                xi,
                y_text,
                f"{val:.2f}",
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                color=colors[row_idx]
            )

    title = metric_name.replace("N_", "").replace("M_", "").replace("_", " ")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_present, rotation=35, ha="right", fontsize=11)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if show_ylabel:
        ax.set_ylabel("Score", fontsize=14, fontweight="bold")

    return fig, ax, True