from GNN_gendata.gen_topo import *


def gen_init_data(current):
    list_of_node, list_of_edge, has_short_cut = convert_graph(current.graph,
                                                              current.comp2port_mapping,
                                                              current.port2comp_mapping,
                                                              current.idx_2_port,
                                                              current.parent,
                                                              current.component_pool,
                                                              current.same_device_mapping,
                                                              current.port_pool)

    G = nx.Graph()
    G.add_nodes_from((list_of_node))
    G.add_edges_from(list_of_edge)
    if nx.is_connected(G) and not has_short_cut:
        G.clear()
        list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(current.graph,
                                                                             current.component_pool,
                                                                             current.port_pool,
                                                                             current.parent,
                                                                             current.comp2port_mapping)
        for i in range(len(list_of_edge)):
            list_of_edge[i] = list(list_of_edge[i])
        prohibit_path = ['VIN - L - GND', 'VIN - L - VOUT', 'VOUT - L - GND',
                         'VIN - Sa - GND', 'VIN - Sb - GND', 'VOUT - Sa - GND', 'VOUT - Sb - GND',
                         'VIN - Sa - Sa - GND', 'VIN - Sb - Sb - GND', 'VOUT - Sa - Sa - GND',
                         'VOUT - Sb - Sb - GND',
                         'VIN - Sa - Sa - Sa - GND', 'VIN - Sb - Sb - Sb - GND',
                         'VOUT - Sa - Sa - Sa - GND', 'VOUT - Sb - Sb - Sb - GND']

        path = find_paths_from_edges(list_of_node, list_of_edge)
        if not check_topo_path(path, prohibit_path) or check_redundant_loop(list_of_node, list_of_edge):
            return None
        # print(path)
        # T = nx.Graph()
        # T.add_nodes_from((list_of_node))
        # T.add_edges_from(list_of_edge)
        # plt.figure()
        # nx.draw(T, with_labels=True)
        #
        # name = "PCC-" + format(num_topology, '06d')
        # file_path = directory_path + '/' + name
        # # print(graph_name)
        # plt.savefig(file_path)  # save as png
        # # plt.show()
        # T.clear()
        # plt.close()

        init_data = {
            "port_2_idx": current.port_2_idx,
            "idx_2_port": current.idx_2_port,
            "port_pool": current.port_pool,
            "component_pool": current.component_pool,
            "same_device_mapping": current.same_device_mapping,
            "comp2port_mapping": current.comp2port_mapping,
            "port2comp_mapping": current.port2comp_mapping,
            "list_of_node": list_of_node,
            "list_of_edge": list_of_edge,
            "netlist": netlist,
            "joint_list": joint_list,
        }
        return init_data
    else:
        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_components', type=int, default=5, help='specify the number of component')
    parser.add_argument('-n_topology', type=int, default=60000,
                        help='specify the number of topology you want to generate')
    parser.add_argument('-output_folder', type=str, default="components_data_random",
                        help='specify the output folder path')

    args = parser.parse_args()

    directory_path = args.output_folder

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    n_components = args.n_components

    k = args.n_topology

    num_topology = 0

    # from datetime import timedelta
    start = time.time()

    data = {}

    while k > 0:
        # print("graph",k)
        component_pool, port_pool, count_map, comp2port_mapping, port2comp_mapping, port_2_idx, idx_2_port, same_device_mapping, graph, parent = initial(
            n_components)  # key is the idx in component pool, value is idx in port pool

        print('finished initialization')

        p1_pool = list(range(len(port_pool)))

        # random.shuffle(p1_pool)

        for cur_point in p1_pool:

            p2_pool = list(range(len(port_pool)))

            random.shuffle(p2_pool)

            if len(graph[cur_point]) > 0 and random.uniform(0, 1) > 0.2:
                # print("not select")
                continue
            for point_2_connect in p2_pool:

                if cur_point == point_2_connect:
                    continue

                if point_2_connect in same_device_mapping and cur_point == same_device_mapping[point_2_connect]:
                    continue  # check ports don't come from same component

                if (cur_point in graph and point_2_connect in graph[cur_point]) or (
                        point_2_connect in graph and cur_point in graph[point_2_connect]):
                    break  # check 2 port are already connected

                # if point_2_connect in graph and (cur_point == 0 or cur_point == 1 or cur_point == 2) and (0 in graph[point_2_connect] or 1 in graph[point_2_connect] or 2 in graph[point_2_connect]):
                #       continue# check 2 port are vout-vin, vin-gnd, ....

                # if cur_point in graph and (point_2_connect == 0 or point_2_connect == 1 or point_2_connect == 2) and (0 in graph[cur_point] or 1 in graph[cur_point] or 2 in graph[cur_point]):
                #       continue# check 2 port are vout-vin, vin-gnd, ....

                root_cur = find(cur_point, parent)
                root_next = find(point_2_connect, parent)
                root0 = find(0, parent)
                root1 = find(1, parent)
                root2 = find(2, parent)

                if root_cur == root_next:
                    continue

                if root_cur == root0 or root_cur == root1 or root_cur == root2:
                    if root0 == root_next or root1 == root_next or root2 == root_next:
                        # print("0,1,2 should not be in the same joint set")
                        continue  # check if cur port is joint set with 0, 1, 2, the other port is also joint set with 0, 1, 2

                graph[cur_point].add(point_2_connect)
                graph[point_2_connect].add(cur_point)
                union(cur_point, point_2_connect, parent)
                # print("found valid")
                break

        # if check_valid(graph):
        if 1 not in graph or 2 not in graph or 0 not in graph:
            # print("there is no 0,1,2")
            continue

        list_of_node, list_of_edge, has_short_cut = convert_graph(graph, comp2port_mapping, port2comp_mapping,
                                                                  idx_2_port, parent, component_pool,
                                                                  same_device_mapping, port_pool)

        G = nx.Graph()
        G.add_nodes_from((list_of_node))
        G.add_edges_from(list_of_edge)

        if nx.is_connected(G) and not has_short_cut:

            G.clear()
            list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(graph, component_pool, port_pool,
                                                                                 parent, comp2port_mapping)

            prohibit_path = ['VIN - L - GND', 'VIN - L - VOUT', 'VOUT - L - GND', 'VIN - Sa - GND', 'VIN - Sb - GND',
                             'VOUT - Sa - GND', 'VOUT - Sb - GND', 'VIN - Sa - Sa - GND', 'VIN - Sb - Sb - GND',
                             'VOUT - Sa - Sa - GND', 'VOUT - Sb - Sb - GND']

            path = find_paths_from_edges(list_of_node, list_of_edge)
            #            print(path)

            if not check_topo_path(path, prohibit_path) or check_redundant_loop(list_of_node, list_of_edge):
                # print('violation: ',path)
                continue
            print(path)
            # print(netlist, joint_list)
            T = nx.Graph()
            T.add_nodes_from((list_of_node))
            T.add_edges_from(list_of_edge)
            # plt.figure(1)
            # nx.draw(G, with_labels=True)
            plt.figure()
            nx.draw(T, with_labels=True)

            name = "PCC-" + format(num_topology, '06d')
            file_path = directory_path + '/' + name
            # print(graph_name)
            plt.savefig(file_path)  # save as png
            # plt.show()
            T.clear()
            plt.close()
            k -= 1

            data[name] = {
                "port_2_idx": port_2_idx,
                "idx_2_port": idx_2_port,
                "port_pool": port_pool,
                "component_pool": component_pool,
                "same_device_mapping": same_device_mapping,
                "comp2port_mapping": comp2port_mapping,
                "port2comp_mapping": port2comp_mapping,
                "list_of_node": list_of_node,
                "list_of_edge": list_of_edge,
                "netlist": netlist,
                "joint_list": joint_list,
            }
            # save_ngspice_file(netlist, file_path, args)
            num_topology += 1

    elapsed_time_secs = time.time() - start

    msg = 'AVG generation took in secs: ', (elapsed_time_secs * 0.1 / num_topology)

    print(msg)

    with open(directory_path + '/data.json', 'w') as outfile:
        json.dump(data, outfile)
