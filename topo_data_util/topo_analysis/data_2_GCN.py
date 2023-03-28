import json
import collections

from topo_analysis.graphUtils import nodes_and_edges_to_adjacency_matrix


def parse_data_json(filename):
    json_file = json.load(open(filename))
    #device_type = {'GND':0, 'VIN':1, 'VOUT':2, 'XL':3, 'MFETA':4, 'MFETB':5, 'XC':6}
    matrices = {}

    for item in json_file:
        list_of_edgs = json_file[item]["list_of_edge"]
        list_of_nodes = json_file[item]["component_pool"]
        node_list = []

        for node in list_of_nodes:
            if not node.isdigit():
                """
                node = node.replace("capacitor-", "XC")
                node = node.replace("inductor-", "XL")
                node = node.replace("FET-A-", "MFETA")
                node = node.replace("FET-B-", "MFETB")
                """
                node_list.append(node)

        assert len(node_list) == 7

        adjacent_matrix = [[0 for i in range(7)] for j in range(7)]

        temp_graph = collections.defaultdict(set)
        graph = collections.defaultdict(set)
        for edge in list_of_edgs:
            # for node in edge:
            """
            edge[0] = edge[0].replace("capacitor-", "XC")
            edge[0] = edge[0].replace("inductor-", "XL")
            edge[0] = edge[0].replace("FET-A-", "MFETA")
            edge[0] = edge[0].replace("FET-B-", "MFETB")
            """
            temp_graph[edge[1]].add(edge[0])

        for key in temp_graph:
            for node1 in temp_graph[key]:
                for node2 in temp_graph[key]:
                    index1 = node_list.index(node1)
                    index2 = node_list.index(node2)
                    graph[index1].add(index2)
                    graph[index2].add(index1)

        for node in graph:
            for nei in graph[node]:
                adjacent_matrix[node][nei] = 1

        """
        cki_file = open("data/" + file_name+'.cki')
        for line in cki_file:
            print(line)
        """

        matrices[item] = {'node_list': node_list, 'matrix': adjacent_matrix}

    return matrices


def parse_data_on_topology(filename):
    json_file = json.load(open(filename))
    matrices = {}

    for item in json_file:
        edge_list = json_file[item]["list_of_edge"]
        node_list = json_file[item]["list_of_node"]

        matrices[item] = {'node_list': node_list, 'edge_list': edge_list}

    return matrices


if __name__ == '__main__':
    parse_data_on_topology('data.json', 'matrices.json')
