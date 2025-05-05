import numpy as np

def high_contrast_colors(n=30, min_diff=16, padding=32):
    
    # Define the range of valid RGB values with padding
    valid_range = np.arange(padding, 256 - padding, min_diff)
    
    # Generate all possible combinations of RGB within the valid range
    possible_colors = np.array(
        [[r, g, b] for r in valid_range for g in valid_range for b in valid_range]
    )
    
    # Shuffle the possible colors to randomize selection
    np.random.shuffle(possible_colors)
    
    # Select the first `n` colors and normalize to [0, 1] range for matplotlib
    selected_colors = possible_colors[:n] / 255.0
    
    # Convert to RGBA by adding an alpha channel
    colors = np.hstack([selected_colors, np.ones((selected_colors.shape[0], 1))])
    
    return colors
