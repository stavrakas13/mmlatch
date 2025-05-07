import matplotlib.pyplot as plt
import numpy as np

def create_histograms(data, title=""):
    """
    Creates a grouped bar chart with 4 categories, each containing the 4 values 
    labeled 'txt', 'au', 'vi', 'all'. Each category is shown in a different colour.
    
    Parameters:
    - data (dict): A dictionary where keys are category names and values are 
                   dictionaries with keys 'txt', 'au', 'vi', 'all'.
                   
    Example:
        data = {
            "Category 1": {"txt": 0.8, "au": 0.75, "vi": 0.85, "all": 0.9},
            "Category 2": {"txt": 0.7, "au": 0.65, "vi": 0.8,  "all": 0.85},
            "Category 3": {"txt": 0.9, "au": 0.85, "vi": 0.95, "all": 0.92},
            "Category 4": {"txt": 0.6, "au": 0.7,  "vi": 0.65, "all": 0.68}
        }
        create_histograms(data)
    """
    # Define the types (the x-axis labels for each histogram)
    types = ["text", "audio", "vision", "all"]
    
    # Extract the category names and number of categories
    categories = list(data.keys())
    n_categories = len(categories)
    n_types = len(types)
    
    # Set positions for the groups (one per type)
    x = np.arange(n_types)
    # Define the width for each bar (adjust if necessary)
    width = 0.2
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use the 'magma' colormap for consistent colours across types
    cmap = plt.get_cmap('magma')
    colours = [cmap(i/(n_types-1)) for i in range(n_types)]
    
    # Plot each category's bars with an offset for grouping
    for i, category in enumerate(categories):
        # Get the values in the order of types: "txt", "au", "vi", "all"
        values = [data[category][t] for t in types]
        # Calculate an offset for the current category
        offset = (i - (n_categories - 1) / 2) * width
        # Create the bar plot for the current category
        ax.bar(x + offset, values, width, label=category, color=colours[i % len(colours)])
    
    # Set the labels and title
    ax.set_ylabel("Accuracy-7")
    ax.set_xlabel("Modalities")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    
    # Adjust y-axis limits to zoom in around 0.4
    ax.set_ylim(0.38, 0.455)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Define example data
    example_data = {
        "Uncorrupted": {"text": 0.434 , "audio": 0.434, "vision": 0.434, "all": 0.434},
        "Gaussian": {"text": 0.432, "audio": 0.427, "vision": 0.434, "all": 0.434},
        "Dropout": {"text": 0.432, "audio": 0.451, "vision": 0.436, "all": 0.415},
        "Shuffle": {"text": 0.415, "audio": 0.436, "vision": 0.421, "all": 0.0},
    }
    example_data_no_mmlatch = {
        "Uncorrupted": {"text": 0.411, "audio": 0.411, "vision": 0.411, "all": 0.411},
        "Gaussian": {"text": 0.411, "audio": 0.411, "vision": 0.411, "all": 0.421},
        "Dropout": {"text": 0.398, "audio": 0.413, "vision": 0.409, "all": 0.396},
        "Shuffle": {"text": 0.402, "audio": 0.417, "vision": 0.419, "all": 0.0},
    }
    title = "Accuracy-7 with noisy Dataset (using MMlatch)"
    title2 = "Accuracy-7 with noisy Dataset (no MMlatch)"
    
    # Create the histograms
    create_histograms(example_data, title)
    create_histograms(example_data_no_mmlatch, title2)
