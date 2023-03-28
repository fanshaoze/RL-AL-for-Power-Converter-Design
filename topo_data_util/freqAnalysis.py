import collections
import json
import math

import criteria
from topo_utils.util import powerset


def frequency_analysis(docs, bag_of_words):
    """
    :return: {word: proportion of docs that contain word}
    """
    doc_num = len(docs)
    result = collections.Counter()
    for word in bag_of_words:
        result[word] = 1. * sum(word in doc for doc in docs) / doc_num

    return result

def tf(doc):
    result = collections.defaultdict(float)
    doc_len = len(doc)
    for word in doc:
        result[word] += 1. / doc_len
    return result

def idf(docs, bag_of_words):
    result = {}
    for word in bag_of_words:
        occur_num = sum(word in doc for doc in docs)
        result[word] = math.log(1. * len(docs) / occur_num)
    return result

def tf_idf_analysis(docs, bag_of_words):
    """
    :return: {word: tf-idf in docs}
    """
    doc_num = len(docs)

    idf_counter = idf(docs, bag_of_words)

    results = collections.defaultdict(float)
    for doc in docs:
        tf_counter = tf(doc)
        for word in doc:
            results[word] += tf_counter[word] * idf_counter[word]

    # normalize
    for word in results.keys():
        results[word] /= 1. * doc_num
        #results[word] /= 1. * doc_occurrence_nums[word]

    return results


def save_stats(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f)

def print_results(results:list):
    for result in results:
        print('path: %s' % result[0])
        print('freq: %.4f' % abs(result[1]))

def print_stats(results):
    print('Most occurrences in positive cases')
    print_results(results[:20])
    print()
    print('Most occurrences in negative cases')
    print_results(list(reversed(results[-20:])))


def compute_pos_node_probs(pos_node_lists, all_nodes=['Sa', 'Sb', 'L', 'C']):
    joint_prob = collections.defaultdict(int)

    all_nodes_with_ids = [node + str(id) for id in range(4) for node in all_nodes]

    for subset in powerset(all_nodes_with_ids):
         for node_list in pos_node_lists:
             if set(subset).issubset(node_list):
                 joint_prob[subset] += 1

    # normalize
    pos_node_list_len = len(pos_node_lists)
    for k, v in joint_prob.items():
        joint_prob[k] = 1. * v / pos_node_list_len

    return joint_prob

def find_most_freq_paths(data, is_positive, is_negative, filename, metric_name='freq'):
    """
    :param data: {name: {paths:, eff:, vout:}}
    :return: [(path, freq), ...] in descent order in freq
    """
    pos_names = []
    pos_docs = []
    neg_docs = []
    bag_of_words = set()

    pos_node_lists = [] # list of nodes of pos topos

    for datum in data:
        name = datum['name']
        paths = datum['paths']
        eff = datum['eff']
        vout = datum['vout']

        if is_positive(eff, vout):
            pos_docs.append(paths)
            pos_names.append(name)

            # nodes_in_paths = list(filter(lambda _: type(_) is str, data[name]['node_list']))
            # nodes_in_paths.remove('VIN')
            # nodes_in_paths.remove('VOUT')
            # nodes_in_paths.remove('GND')
            #
            # pos_node_lists.append(nodes_in_paths)
        elif is_negative(eff, vout):
            neg_docs.append(paths)
        else:
            # not gooing to deal with moderate topos
            continue

        bag_of_words.update(paths)

    print('positive topos:', pos_names)

    if metric_name == 'freq':
        metric = frequency_analysis
    elif metric_name == 'tfidf':
        metric = tf_idf_analysis
    else:
        raise Exception('unknown metric ' + metric_name)

    #pos_node_probs = compute_pos_node_probs(pos_node_lists)
    #print(pos_node_probs)
    #save_stats(list(pos_node_probs.items()), filename + '_node_joint_probs.json')

    pos_results = metric(pos_docs, bag_of_words)
    neg_results = metric(neg_docs, bag_of_words)
    print('positive cases', len(pos_docs))
    print('negative cases', len(neg_docs))

    results = pos_results
    results.subtract(neg_results)
    # [(path, freq), ...] in descent order in freq
    results = list(sorted(results.items(), key=lambda _: _[1], reverse=True))

    #paths = [result[0] for result in results]
    #visualize_paths(paths[:5], 'most_freq_paths')
    #visualize_paths(reversed(paths[-5:]), 'least_freq_paths')

    save_stats(results, filename + '_path_freqs.json')
    print_stats(results)


def process(filename):
    data = json.load(open(filename + '.json'))
    # TODO: must specify the target vout for the high_reward and low_reward in the criteria
    find_most_freq_paths(data, criteria.high_reward, criteria.low_reward, filename)

if __name__ == '__main__':
    process('dataset_5_all')
