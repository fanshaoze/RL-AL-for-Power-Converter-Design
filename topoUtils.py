from topo_data_util.topo_analysis.topoGraph import TopoGraph


def find_paths(state):
    """
    Useful for GP and transformer based surrogate model
    Return the list of paths in the current state
    e.g. ['VIN - inductor - VOUT', ...]
    """
    node_list, edge_list = state.get_nodes_and_edges()

    # convert graph to paths, and find embedding
    topo = TopoGraph(node_list=node_list, edge_list=edge_list)
    return topo.find_end_points_paths_as_str()


def find_paths_with_topo_info(node_list, edge_list):
    """
    Useful for GP and transformer based surrogate model with list of node and edge
    Return the list of paths in the current state
    e.g. ['VIN - inductor - VOUT', ...]
    """
    # node_list, edge_list = state.get_nodes_and_edges()
    # convert graph to paths, and find embedding
    topo = TopoGraph(node_list=node_list, edge_list=edge_list)
    return topo.find_end_points_paths_as_str()


def isomorphize_state_list(state_list):
    """
    Clean same topologies by path list. Useful for active learning selection.
    """
    paths_list = set()
    cleaned_states = []

    for state in state_list:
        # only consider potentially valid topos
        if state.graph_is_valid():
            paths = tuple(find_paths(state))

            if paths not in paths_list:
                paths_list.add(paths)
                cleaned_states.append(state)

    return cleaned_states

def get_topo_key(state):
    """
    Get a unique string that represents a state
    """
    return state.get_key() + '$' + str(state.parameters)
