from topo_analysis.topoGraph import TopoGraph


def remove_redundant_loop_from_edges(node_list, edge_list):
    """
    Return the new node list and edge list with redundant loops removed
    """
    topo_graph = TopoGraph(node_list=node_list, edge_list=edge_list)
    topo_graph.eliminate_redundant_comps()

    new_node_list = topo_graph.get_node_list()
    new_edge_list = topo_graph.get_edge_list()
    return new_node_list, new_edge_list

# deprecated names used before
remove_redundant_loop_on_edges = remove_redundant_loop_from_edges


if __name__ == '__main__':
    """
    A simple example:

           GND
          /    \
        VIN -- VOUT
          \   /
            x  -- y

    'y' should be eliminated.
    """
    node_list = ['VIN', 'VOUT', 'GND', 'x', 'y']
    edge_list = [('VIN', 'VOUT'), ('VIN', 'GND'), ('VIN', 'x'), ('VOUT', 'VIN'), ('VOUT', 'GND'), ('VOUT', 'x'),
                 ('GND', 'VIN'), ('GND', 'VOUT'), ('x', 'VIN'), ('x', 'VOUT'), ('x', 'y'), ('y', 'x')]

    print(remove_redundant_loop_from_edges(node_list, edge_list))
    # output: (['VIN', 'VOUT', 'GND', 'x'],
    #         [('VIN', 'VOUT'), ('VIN', 'GND'), ('VIN', 'x'), ('VOUT', 'VIN'), ('VOUT', 'GND'), ('VOUT', 'x'), ('GND', 'VIN'), ('GND', 'VOUT'), ('x', 'VIN'), ('x', 'VOUT')])
