from GNN_gendata.gen_topo import *


def get_key_data(init_data):
    json_file = {'one_topo': init_data}
    key_list = key_circuit('one_topo', json_file)
    key = key_list[0]

    key_data = {
        "key": key,
        "list_of_node": init_data['list_of_node'],
        "list_of_edge": init_data['list_of_edge'],
        "netlist": init_data['netlist']
    }
    return key, key_data


if __name__ == '__main__':

    json_file = json.load(open("./components_data_random/data.json"))

    data_folder = 'component_data_random'
    target_folder = 'database'

    circuit_dic = {}

    data = {}

    for fn in json_file:

        tmp = json_file

        key_list = []

        key_list = key_circuit(fn, tmp)

        for key in key_list:
            if key in circuit_dic:
                circuit_dic[key].append(fn)
            else:
                circuit_dic[key] = []
                circuit_dic[key].append(fn)

    count = 0;

    with open(target_folder + '/key.json', 'w') as outfile:
        json.dump(circuit_dic, outfile)
    outfile.close()

    filename_list = []

    json_file = json.load(open("./components_data_random/data.json"))

    for key in circuit_dic:

        filename = circuit_dic[key][0]

        if filename not in filename_list:
            filename_list.append(filename)
        else:
            continue

        list_of_node = json_file[filename]['list_of_node']
        list_of_edge = json_file[filename]['list_of_edge']
        netlist = json_file[filename]['netlist']

        #            print(netlist)

        name = 'Topo-' + format(count, '06d')
        topo_file = target_folder + '/topo/' + name + '.png'

        save_topo(list_of_node, list_of_edge, topo_file)

        count = count + 1

        Original_name = circuit_dic[key][0]

        print(name, Original_name)

        data[name] = {
            "key": key,
            "File_name": Original_name,
            "list_of_node": list_of_node,
            "list_of_edge": list_of_edge,
            "netlist": netlist
        }

    with open(target_folder + '/data.json', 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()

    print(len(data))
