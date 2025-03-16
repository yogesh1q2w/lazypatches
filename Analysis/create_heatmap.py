import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_heatmap(hist_distances, is_correct):
    hist_distances = np.array(hist_distances)
    is_correct = np.array(is_correct)
    bins = np.linspace(0, 1, 11)  # 11 intervals, 12 bin edges
    
    hist_success, _ = np.histogram(hist_distances[is_correct], bins=bins)
    hist_failure, _ = np.histogram(hist_distances[~is_correct], bins=bins)
    
    heatmap_success = hist_success[np.newaxis, :]
    heatmap_failure = hist_failure[np.newaxis, :]

    print(hist_success)
    print(heatmap_failure)
    
    green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['lightgreen', 'darkgreen'])
    red_cmap = LinearSegmentedColormap.from_list('red_cmap', ['lightcoral', 'darkred'])
    
    xticklabels = np.round(bins, 1).tolist()  # Corrected xticklabels
    
    fig, axs = plt.subplots(2, figsize=(11, 2))  # Adjusted size to fit the heatmaps vertically


    # Plot success heatmap on the first subplot
    sns.heatmap(heatmap_success, annot=True, fmt='.1f', cmap=green_cmap,
                yticklabels=['Success'], xticklabels=xticklabels, linewidths=0.5, ax=axs[0])
    axs[0].set_xticklabels(xticklabels, rotation=45, ha="right")  # Rotate and align x-ticks
    axs[0].set_xticks(np.arange(len(xticklabels)))  # Center x-ticks correctly
    axs[0].tick_params(axis='x', which='both', length=0)  # Remove x-axis ticks

    # Plot failure heatmap on the second subplot
    sns.heatmap(heatmap_failure, annot=True, fmt='.1f', cmap=red_cmap, 
                yticklabels=['Failure'], xticklabels=xticklabels, linewidths=0.5, ax=axs[1])
    plt.xticks(ticks=np.arange(len(xticklabels)), labels=xticklabels)  # Align x-ticks properly


    plt.subplots_adjust(hspace=0) 

    plt.show()

def main():
    np.random.seed(42)
    hist_distances = np.random.uniform(0, 1, 1500)
    is_correct = np.random.choice([True, False], size=1500)
    create_heatmap(hist_distances, is_correct)

if __name__ == "__main__":
    main()
