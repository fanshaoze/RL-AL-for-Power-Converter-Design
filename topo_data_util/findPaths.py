from topo_data_util.topo_analysis.topoGraph import TopoGraph


def find_paths_from_edges(node_list, edge_list):
    return TopoGraph(node_list=node_list, edge_list=edge_list).find_end_points_paths_as_str()


def find_paths_from_adj_matrix(node_list, adj_matrix):
    return TopoGraph(node_list=node_list, adj_matrix=adj_matrix).find_end_points_paths_as_str()


if __name__ == '__main__':
    """
    A simple example:

                GND
                 | 
        VIN -- FET-A  -- VOUT
        
    """
    node_list = ['VIN', 'VOUT', 'GND', 'FET-A']
    # edges are bidirectional
    edge_list = [('VIN', 'FET-A'), ('FET-A', 'GND'), ('FET-A', 'VOUT')]

    print(find_paths_from_edges(node_list, edge_list))
    # output: ['VIN - FET-A - VOUT', 'VIN - FET-A - GND', 'VOUT - FET-A - GND']
