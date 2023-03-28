from SimulatorAnalysis.gen_topo import *


def generate_key_data(directory_path, name, target_folder):
    json_file = json.load(open(directory_path + '/' + name + '-data.json'))

    # target_folder = 'database'

    circuit_dic = {}

    data = {}

    for fn in json_file:
        tmp = json_file
        key_list = []
        key_list = key_circuit(fn, tmp)
        tmp_key_list= []
        # print("len key list:", len(key_list))
        for key in key_list:
            # print(key)
            if key not in tmp_key_list:
                tmp_key_list.append(key)
        # print('tmp_key_list:',len(tmp_key_list))

        for key in key_list:
            if key in circuit_dic:
                circuit_dic[key].append(fn)
            else:
                circuit_dic[key] = []
                circuit_dic[key].append(fn)

    count = 0

    with open(target_folder + '/key.json', 'w') as outfile:
        json.dump(circuit_dic, outfile)
    outfile.close()

    filename_list = []

    json_file = json.load(open("./" + directory_path + '/' + name + '-data.json'))

    for key in circuit_dic:

        # print(key, count)
        filename = circuit_dic[key][0]

        if filename not in filename_list:
            filename_list.append(filename)
        else:
            continue

        list_of_node = json_file[filename]['list_of_node']
        list_of_edge = json_file[filename]['list_of_edge']
        netlist = json_file[filename]['netlist']

        #            print(netlist)

        topo_file = target_folder + '/topo/' + name + '.png'

        # save_topo(list_of_node, list_of_edge, topo_file)

        count = count + 1

        data[name] = {
            "key": key,
            "list_of_node": list_of_node,
            "list_of_edge": list_of_edge,
            "netlist": netlist
        }

    with open(target_folder + '/' + name + '-data.json', 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()

    print(len(data))
    print(key)
    return key


if __name__ == '__main__':

    json_file = json.load(open("./components_data_random/data.json"))

    data_folder = 'component_data_random'
    target_folder = 'database'

    circuit_dic = {}

    data = {}

    for fn in json_file:

        tmp = json_file

        key = key_circuit(fn, tmp)

        if key in circuit_dic:
            circuit_dic[key].append(fn)
        else:
            circuit_dic[key] = []
            circuit_dic[key].append(fn)

    count = 0;

    json_file = json.load(open("./components_data_random/data.json"))

    for key in circuit_dic:
        filename = circuit_dic[key][0]

        list_of_node = json_file[filename]['list_of_node']
        list_of_edge = json_file[filename]['list_of_edge']
        netlist = json_file[filename]['netlist']

        # print(netlist)

        name = 'Topo-' + format(count, '04d')
        topo_file = target_folder + '/topo/' + name + '.png'

        save_topo(list_of_node, list_of_edge, topo_file)

        count = count + 1

        data[name] = {
            "key": key,
            "list_of_node": list_of_node,
            "list_of_edge": list_of_edge,
            "netlist": netlist
        }

    with open(target_folder + '/data.json', 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()

    # print(len(data))
