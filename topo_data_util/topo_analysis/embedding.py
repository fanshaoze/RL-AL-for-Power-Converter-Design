import json
import logging
import os

import numpy as np

from freqAnalysis import tf, idf, tf_idf_analysis

def tf_idf_embed(paths, idf_counter, vec_of_paths):
    tf_counter = tf(paths)
    return tuple([tf_counter[path] * idf_counter[path] for path in vec_of_paths])

def tf_idf_uniform_embed(paths, tf_idf_counter, vec_of_paths):
    return tuple([(tf_idf_counter[path] if path in paths else 0) for path in vec_of_paths])

def tf_embed(paths, vec_of_paths):
    tf_counter = tf(paths)
    return tuple([tf_counter[path] for path in vec_of_paths])

device_list = ['GND', 'VIN', 'VOUT', 'capacitor', 'FET-A', 'FET-B', 'inductor']
def vector_embed(paths, vec_of_paths):
    output = []
    for path in vec_of_paths:
        if path in paths:
            output.append([1 * (device in path) for device in device_list])
        else:
            output.append([0] * len(device_list))

    return output

def boolean_embed(paths, vec_of_paths):
    tf_counter = tf(paths)
    return tuple([tf_counter[path] > 0 for path in vec_of_paths])

def embed_data(filename, scale=1., size_needed=None, use_names=None, key='eff', embed='freq'):
    """
    data should contain paths.
    :return: training data (embedding and efficiencies), the embedding function
    """
    data = json.load(open(filename + '.json', 'r'))

    bag_of_paths = set()

    if use_names is not None:
        names = use_names
    else:
        names = list(data.keys())

    if size_needed:
        names = names[:size_needed]

    for name in names:
        bag_of_paths.update(data[name]['paths'])

    vec_of_paths = sorted(list(bag_of_paths))
    logging.debug('len of bag of paths ' + str(len(vec_of_paths)))

    # create dataset
    dataset_x = []
    dataset_y = []

    # pre-compute tfidf counts
    if embed == 'tfidf':
        idf_counter = idf([data[name]['paths'] for name in names], vec_of_paths)
    elif embed == 'tfidf-uniform':
        tf_idf_counter = tf_idf_analysis([data[name]['paths'] for name in names], vec_of_paths)

    for name in names:
        paths = data[name]['paths']

        if embed == 'tfidf':
            x = tf_idf_embed(paths, idf_counter, vec_of_paths)
        elif embed == 'tfidf-uniform':
            x = tf_idf_uniform_embed(paths, tf_idf_counter, vec_of_paths)
        elif embed == 'freq':
            x = tf_embed(paths, vec_of_paths)
        elif embed == 'boolean':
            x = boolean_embed(paths, vec_of_paths)
        elif embed == 'vector':
            x = vector_embed(paths, vec_of_paths)
        else:
            raise Exception('unknown embed method ' + str(embed))

        y = data[name][key] * 1. / scale

        """
        for idx in range(len(dataset_x)):
            if dataset_x[idx] == x and dataset_y[idx] != y:
                logging.debug('different y ' + str(y) + ' ' + str(dataset_y[idx]))
        """

        dataset_x.append(x)
        dataset_y.append(y)

    logging.debug('total data loaded ' + str(len(dataset_x)))

    return np.array(names), np.array(dataset_x), np.array(dataset_y), vec_of_paths


if __name__ == '__main__':
    # todo should implement 'finding embedding and save' here, not in gpytorch
    pass