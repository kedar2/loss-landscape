import matplotlib.pyplot as plt
import matplotlib
import csv

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

def plot_dict_values(d):
    """
    Creates separate plots of the values for each key in a dictionary.
    """

    for key, value in d.items():
        fig, ax = plt.subplots()
        ax.plot(value, label=key)
        ax.set_title(key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        plt.show()