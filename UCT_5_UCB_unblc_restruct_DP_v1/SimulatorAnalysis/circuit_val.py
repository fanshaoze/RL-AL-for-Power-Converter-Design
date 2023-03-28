from SimulatorAnalysis.gen_topo import *


def assign_C_and_L_in_param(param, fix_paras):
    assert fix_paras['C'] != []
    assert fix_paras['L'] != []
    param['C'] = fix_paras['C']
    param['L'] = fix_paras['L']
    return param


def get_one_circuit(target_folder, name, fix_paras, pass_para=True):
    json_file = json.load(open(target_folder + '/' + name + '-data.json'))
    data = {}
    param = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    param = assign_C_and_L_in_param(param, fix_paras)
    print(param)

    for fn in json_file:

        net_list = json_file[fn]['netlist']
        device_list = []
        node_list = []

        # circuit_a = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 100\n' + 'Cout OUT 0\n'
        # circuit_b = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 100\n' + 'Cout OUT 0\n'
        circuit_a = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 50\n' + 'Cout OUT 0\n'
        circuit_b = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 50\n' + 'Cout OUT 0\n'

        device_list = ['Vin', 'Rin', 'Rout', 'Cout']

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
            if pass_para:
                if device[0] == 'C':
                    line_a = line_a + ' ' + str(param['C'][0])
                    line_b = line_b + ' ' + str(param['C'][0])
                if device[0] == 'L':
                    line_a = line_a + ' ' + str(param['L'][0])
                    line_b = line_b + ' ' + str(param['L'][0])
                """Possible cases"""
                if device[0:2] == 'Sa':
                    line_a = line_a + ' ' + str(param['Ra'][0])
                    line_b = line_b + ' ' + str(param['Rb'][0])
                if device[0:2] == 'Sb':
                    line_a = line_a + ' ' + str(param['Rb'][0])
                    line_b = line_b + ' ' + str(param['Ra'][0])

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

    with open(target_folder + '/' + name + '-circuit_val.json', 'w') as outfile:
        json.dump(data, outfile)
    return data[fn]


if __name__ == '__main__':

    json_file = json.load(open("./database/data.json"))

    target_folder = './database'

    data = {}

    for fn in json_file:

        net_list = json_file[fn]['netlist']
        device_list = []
        node_list = []

        circuit_a = 'Vin IN_exact 0\nRin IN_exact IN\n' + 'Rout OUT 0\n' + 'Cout OUT 0\n'
        circuit_b = 'Vin IN_exact 0\nRin IN_exact IN\n' + 'Rout OUT 0\n' + 'Cout OUT 0\n'

        device_list = ['Vin', 'Rin', 'Rout', 'Cout']

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
