
from topo_data_util.topo_analysis.graphUtils import indexed_graph_to_adjacency_matrix, adj_matrix_to_graph, graph_to_adjacency_matrix, \
    nodes_and_edges_to_adjacency_matrix, adj_matrix_to_edges


class TopoGraph(object):
    def __init__(self, node_list, adj_matrix=None, graph=None, edge_list=None, hint=None):
        self.node_list = node_list

        if adj_matrix is not None:
            self.adj_matrix = adj_matrix
        elif graph is not None:
            if hint == 'indexed':
                self.adj_matrix = indexed_graph_to_adjacency_matrix(graph)
            else:
                self.adj_matrix = graph_to_adjacency_matrix(graph, node_list)
        elif edge_list is not None:
            self.adj_matrix = nodes_and_edges_to_adjacency_matrix(node_list, edge_list)
        else:
            raise Exception('failed to initialize Graph')

    def modify_port(self, port: str) -> str:
        """
        Merge ports for topo_analysis
        """
        if port.startswith('L'):
            return 'L'
        elif port.startswith('C'):
            return 'C'
        elif port.startswith('Sa'):
            return 'Sa'
        elif port.startswith('Sb'):
            return 'Sb'
        # for old representation
        elif port.startswith('inductor'):
            return 'inductor'
        if port.startswith('capacitor'):
            return 'capacitor'
        if port.startswith('FET-A'):
            return 'FET-A'
        elif port.startswith('FET-B'):
            return 'FET-B'

        # keep as is
        return port

    def find_paths(self, source: int, target: int, exclude=[]) -> list:
        """
        Return a list of paths from source to target without reaching `exclude`.

        :param adj_matrix: the adjacency matrix of a graph
        :param exclude: nodes in this list are excluded from the paths (e.g. VIN to VOUT *without* reaching GND)
        """
        node_num = len(self.adj_matrix)

        paths = []

        def dfs(s, t, cur_path):
            """
            Perform dfs starting from s to find t, excluding nodes in exclude.
            cur_path stores the node visited on the current path.
            Results are added to paths.
            """
            if s in exclude:
                return

            if s == t:
                paths.append(cur_path + [s])
                return

            for neighbor in range(node_num):
                # find neighbors that are not visited in this path
                if neighbor != s and self.adj_matrix[s][neighbor] == 1 and not neighbor in cur_path:
                    dfs(neighbor, t, cur_path + [s])

        dfs(source, target, [])

        return paths

    def find_end_points_paths(self):
        """
        Find paths between any of VIN, VOUT, GND
        """
        gnd = self.node_list.index('GND')
        vin = self.node_list.index('VIN')
        vout = self.node_list.index('VOUT')

        paths = self.find_paths(vin, vout, [gnd]) + \
                self.find_paths(vin, gnd, [vout]) + \
                self.find_paths(vout, gnd, [vin])

        return paths

    def encode_path_as_string(self, path):
        """
        Convert a path to a string, so it's hashbale and readable
        """
        # 1. convert to node list
        path = [self.node_list[idx] for idx in path]

        # 2. drop connection nodes
        path = list(filter(lambda port: not isinstance(port, int), path))

        # 3. merge ports with different ids
        path = [self.modify_port(port) for port in path]

        # 4. to string
        path = ' - '.join(path)

        return path

    def find_end_points_paths_as_str(self):
        paths = self.find_end_points_paths()
        paths_str = [self.encode_path_as_string(path) for path in paths]

        return paths_str

    def eliminate_redundant_comps(self):
        """
        Remove redundant components in the adjacency matrix.
        """
        node_num = len(self.node_list)
        paths = self.find_end_points_paths()

        # compute traversed nodes
        traversed_nodes = set()
        for path in paths:
            traversed_nodes.update(path)
        traversed_nodes = list(traversed_nodes)

        new_matrix =\
            [[self.adj_matrix[i][j] for j in range(node_num) if j in traversed_nodes]
                                    for i in range(node_num) if i in traversed_nodes]
        new_node_list = [self.node_list[idx] for idx in traversed_nodes]

        self.adj_matrix = new_matrix
        self.node_list = new_node_list

    def get_graph(self):
        return adj_matrix_to_graph(self.node_list, self.adj_matrix)

    def get_edge_list(self):
        return adj_matrix_to_edges(self.node_list, self.adj_matrix)

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_node_list(self):
        return self.node_list


