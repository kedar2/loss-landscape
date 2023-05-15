import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm

def export_dict_to_csv(d, filename):
    """
    Exports a dictionary to a csv file.

    Args:
        d (dict): Dictionary to export.
        filename (str): Name of the csv file.
    """

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())

        for row in zip(*d.values()):
            writer.writerow(row)

def plot_dict_values(d, directory='results/'):
    """
    Creates separate plots of the values for each key in a dictionary.

    Args:
        d (dict): Dictionary to plot.
        directory (str): Path to save the plots to.
    """

    for key, value in d.items():
        fig, ax = plt.subplots()
        ax.plot(value, label=key)
        ax.set_title(key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        plt.savefig(directory + key + '.png')
        plt.show()

def generate_heatmap(x_range: list, y_range: list, f: callable, xlabel: str='x', ylabel: str='y', title: str='f(x, y)'):
    """
    Generates a heatmap of the function f over the range specified by x_range and y_range.

    Args:
        x_range (list): List of x values to evaluate f at.
        y_range (list): List of y values to evaluate f at.
        f (callable): Function to evaluate.
    """

    f_values = np.zeros((len(y_range), len(x_range)))
    for i, x in tqdm(enumerate(x_range)):
        for j, y in enumerate(y_range):
            f_values[j, i] = f(x, y)
    ax = sns.heatmap(f_values, xticklabels=x_range, yticklabels=y_range, cmap='viridis')
    ax.invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()