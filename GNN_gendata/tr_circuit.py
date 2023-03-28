from GNN_gendata.gen_topo import *


def get_circuit_data(key_data):
    net_list = key_data['netlist']
    circuit_a = 'Vin IN_exact 0 1\nRin IN_exact IN 1\n' + 'Rout OUT 0 {Rout}\n' + 'Cout OUT 0 10e-6\n'
    circuit_b = 'Vin IN_exact 0 1\nRin IN_exact IN 1\n' + 'Rout OUT 0 {Rout}\n' + 'Cout OUT 0 10e-6\n'

    device_list = ['Vin', 'Rin', 'Rout']

    node_list = ['IN', 'OUT', 'IN_exact']

    for item in net_list:
        line_a = ''
        line_b = ''
        device = item[0]

        if device[0:2] == 'Sa':
            line_a = 'Ra' + device[2:]
            line_b = 'Rb' + device[2:]
            if line_a not in device_list:
                device_list.append(line_a)
            if line_b not in device_list:
                device_list.append(line_b)
        elif device[0:2] == 'Sb':
            line_a = 'Rb' + device[2:]
            line_b = 'Ra' + device[2:]
            if line_a not in device_list:
                device_list.append(line_a)
            if line_b not in device_list:
                device_list.append(line_b)
        else:
            line_a = device
            device_list.append(line_a)
            line_b = device

        for node in item[1::]:
            line_a = line_a + ' ' + node
            line_b = line_b + ' ' + node
            if node not in node_list:
                node_list.append(node)
        line_a = line_a + '\n'
        line_b = line_b + '\n'
        circuit_a = circuit_a + line_a
        circuit_b = circuit_b + line_b

    circuit_data = {"key": key_data['key'],
                    "circuit_a": circuit_a,
                    "circuit_b": circuit_b,
                    "device_list": device_list,
                    "node_list": node_list,
                    "net_list": net_list
                    }
    return circuit_data


if __name__ == '__main__':

    json_file = json.load(open("./database/data.json"))

    target_folder = './database'

    data = {}

    for fn in json_file:

        net_list = json_file[fn]['netlist']
        device_list = []
        node_list = []

        circuit_a = 'Vin IN_exact 0 1\nRin IN_exact IN 1\n' + 'Rout OUT 0 {Rout}\n' + 'Cout OUT 0 10e-6\n'
        circuit_b = 'Vin IN_exact 0 1\nRin IN_exact IN 1\n' + 'Rout OUT 0 {Rout}\n' + 'Cout OUT 0 10e-6\n'

        device_list = ['Vin', 'Rin', 'Rout']

        node_list = ['IN', 'OUT', 'IN_exact']

        for item in net_list:
            line_a = ''
            line_b = ''
            device = item[0]

            if device[0:2] == 'Sa':
                line_a = 'Ra' + device[2:]
                line_b = 'Rb' + device[2:]
                if line_a not in device_list:
                    device_list.append(line_a)
                if line_b not in device_list:
                    device_list.append(line_b)
            elif device[0:2] == 'Sb':
                line_a = 'Rb' + device[2:]
                line_b = 'Ra' + device[2:]
                if line_a not in device_list:
                    device_list.append(line_a)
                if line_b not in device_list:
                    device_list.append(line_b)
            else:
                line_a = device
                device_list.append(line_a)
                line_b = device

            for node in item[1::]:
                line_a = line_a + ' ' + node
                line_b = line_b + ' ' + node
                if node not in node_list:
                    node_list.append(node)
            line_a = line_a + '\n'
            line_b = line_b + '\n'
            circuit_a = circuit_a + line_a
            circuit_b = circuit_b + line_b

        data[fn] = {"key": json_file[fn]['key'],
                    "circuit_a": circuit_a,
                    "circuit_b": circuit_b,
                    "device_list": device_list,
                    "node_list": node_list,
                    "net_list": net_list
                    }

    with open(target_folder + '/circuit.json', 'w') as outfile:
        json.dump(data, outfile)
