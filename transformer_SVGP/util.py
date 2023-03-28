import os
import random

import numpy as np
import torch
import networkx as nx
from matplotlib import pyplot as plt


def feed_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for reproducibility of LSTM
    # see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:2'


def compute_rse(pred, y):
    """
    :param pred: model predictions, a numpy array
    :param y: ground truth,  a numpy array
    """
    pred = np.array(pred)
    y = np.array(y)

    model_mse = np.mean(np.square(pred - y), axis=0)

    y_mean = np.mean(y, axis=0)
    baseline_mse = np.mean(np.square(y - y_mean), axis=0)

    return model_mse / baseline_mse


def compute_classification_error(pred, y):
    binary_pred = (pred >= 0.5)

    tp = np.sum(np.logical_and(binary_pred == 1, y == 1))
    tn = np.sum(np.logical_and(binary_pred == 0, y == 0))

    fp = np.sum(np.logical_and(binary_pred == 1, y == 0))
    fn = np.sum(np.logical_and(binary_pred == 0, y == 1))

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    error = np.mean(binary_pred != y)

    return error, {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'TPR': tpr, 'TNR': tnr}


def visualize(list_of_node, list_of_edge, info=None, filename=None):
    """
    plot circuit topology
    """
    T = nx.Graph()
    T.add_nodes_from(list_of_node)
    T.add_edges_from(list_of_edge)

    pos = nx.nx_agraph.graphviz_layout(T)
    nx.draw(T, pos=pos, with_labels=True)

    if info is not None:
        # write info in the bottom-left corner of fig
        plt.figtext(0.02, 0.02, info)

    plt.savefig(filename + '.png')
    plt.close()


def distribution_plot(simulation, predictions, file_name, rse=0.0):
    """
    plot the distribution
    @param rse:
    @param simulation:
    @param predictions: the predictions of different models
    @param file_name:
    @return:
    """
    # Use a breakpoint in the code line below to debug your script.
    x = simulation
    y = simulation

    #
    # plt.xlabel("evaluation")
    # plt.ylabel("prediction")
    # plt.plot(x, y, 'ob', color='b', markersize='2')
    # plt.plot(x, y_1, 'ob', color='y', markersize='2')
    plt.scatter(x=x, y=y, linewidths=0, c='b', alpha=0.2, s=10)
    for y_1 in predictions:
        plt.scatter(x=x, y=y_1, linewidths=0, c='y', alpha=0.2, s=10)
    plt.title("rse=" + str(rse))
    # plt.yscale('log', base=2)
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(file_name, dpi=1200)
    plt.close()
