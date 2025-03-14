import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_heatmap(hist_distances, is_correct):

    hist_distances = np.array(hist_distances)
    is_correct = np.array(is_correct)
    bins = np.linspace(0, 1, 11)
    hist_success, _ = np.histogram(hist_distances[is_correct], bins=bins)
    hist_failure, _ = np.histogram(hist_distances[~is_correct], bins=bins)
    heatmap_success = hist_success[:, np.newaxis]
    heatmap_failure = hist_failure[:, np.newaxis]

    green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['lightgreen', 'darkgreen'])
    red_cmap = LinearSegmentedColormap.from_list('red_cmap', ['lightcoral', 'darkred'])

    yticklabels = [f'{i:.1f}-{j:.1f}' for i, j in zip(bins[:-1], bins[1:])][::-1]

    # Plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(heatmap_success[::-1], annot=True, fmt='d', cmap=green_cmap, cbar_kws={'label': 'Count'},
                xticklabels=['Success'], yticklabels=yticklabels, linewidths=0.5)
    plt.title('Heatmap(Success)')
    plt.show()
    plt.figure(figsize=(6, 6))
    sns.heatmap(heatmap_failure[::-1], annot=True, fmt='d', cmap=red_cmap, cbar_kws={'label': 'Count'},
                xticklabels=['Failure'], yticklabels=yticklabels, linewidths=0.5)
    plt.title('Heatmap(Failure)')
    plt.show()

#create_heatmap(hist_distances, is_correct)
