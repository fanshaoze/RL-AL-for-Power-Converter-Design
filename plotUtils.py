import torch

from matplotlib import pyplot as plt
import numpy as np


def plot_errorbars(x_ticks, x_values, y_values, y_errors, x_label, y_label, title, filename):
    if x_values is None:
        x_values = range(y_values)

    plt.figure(figsize=(4, 3))

    plt.bar(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim([min(x_values) - 1, max(x_values) + 1])
    plt.ylim([0, 2])

    if x_ticks is not None:
        plt.xticks(x_values, x_ticks)

    plt.tight_layout()

    plt.savefig(filename + ".png", dpi=300, format="png")

    plt.close()

def plot_hist(values, x_label, title, filename):
    plt.figure(figsize=(4, 3))

    plt.hist(values.numpy(), bins=np.arange(-20, 21) * .05)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(filename + ".pdf", dpi=300, format="pdf")

    plt.close()

def bar_plot(x_values, y_values, x_label, y_label, filename):
    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        plt.bar(x_values, y_values)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.savefig(filename + ".pdf", dpi=300, format="pdf")

        plt.close()

def plot_heatmap(grid, filename, extent=None, points=None, path=None, query_data_size=0):
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    plt.imshow(grid, extent=extent)
    plt.colorbar()

    if points is not None:
        data_size = points.size()[0]
        # training data
        plt.plot(points[:data_size - query_data_size, 0], points[:data_size - query_data_size, 1], 'k*')
        # queries
        plt.plot(points[data_size - query_data_size:, 0], points[data_size - query_data_size:, 1], 'r*')

    if path is not None:
        plt.plot(path[:, 0], path[:, 1], 'b')

    plt.savefig(filename + ".png", dpi=300, format="png")

    plt.close()


