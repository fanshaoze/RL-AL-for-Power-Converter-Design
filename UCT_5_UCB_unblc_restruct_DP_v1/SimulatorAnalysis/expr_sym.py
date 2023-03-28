from lcapy import Circuit
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import json


def get_one_expression(target_folder, name):
    json_file = json.load(open(target_folder + '/' + name + '-circuit_sym.json'))

    invalid_topo = []

    data = {}

    for fn in json_file:

        circuit_a = json_file[fn]['circuit_a'].split('\n')
        circuit_b = json_file[fn]['circuit_b'].split('\n')
        device_list = json_file[fn]['device_list']
        node_list = json_file[fn]['node_list']
        net_list = json_file[fn]['net_list']

        cct_a = Circuit()
        cct_b = Circuit()

        # print(circuit_a)
        # print(circuit_b)

        for item in circuit_a:
            #            if item !='' and item[0]=='V':
            cct_a.add(item)

        for item in circuit_b:
            #            if item !='' and item[0]=='V':
            cct_b.add(item)

        # print(net_list)
        # print(cct_a)
        # print(cct_b)
        # print(device_list)

        #        Vin, Rin, Rout, Rb0, Ra0, Rb1, Ra1, Rb2, Ra2 = symbols('Vin Rin Rout Rb0 Ra0 Rb1 Ra1 Rb2 Ra2')
        #        for item in device_list:
        #               vars()[item]=symbols(item)

        data[fn] = {}

        try:
            ss_a = cct_a.ss
            ss_b = cct_b.ss
        except:
            invalid_topo.append(json_file[fn]['key'])
            print("%s violations\n" % fn)
            return "invalid"

        a_X = str(ss_a.x)[7:-1]
        a_Y = str(ss_a.y)[7:-1]
        a_A = str(ss_a.A)[7:-1]
        a_B = str(ss_a.B)[7:-1]
        a_C = str(ss_a.C)[7:-1]
        a_D = str(ss_a.D)[7:-1]

        a = {'x': a_X,
             'y': a_Y,
             'a': a_A,
             'b': a_B,
             'c': a_C,
             'd': a_D
             }

        b_X = str(ss_b.x)[7:-1]
        b_Y = str(ss_b.y)[7:-1]
        b_A = str(ss_b.A)[7:-1]
        b_B = str(ss_b.B)[7:-1]
        b_C = str(ss_b.C)[7:-1]
        b_D = str(ss_b.D)[7:-1]

        b = {'x': b_X,
             'y': b_Y,
             'a': b_A,
             'b': b_B,
             'c': b_C,
             'd': b_D
             }

        data[fn] = {'A state': a,
                    'B state': b,
                    'device_list': device_list,
                    'node_list': node_list,
                    'net_list': net_list}

    with open(target_folder + '/' + name + '-expr_sym.json', 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()

    with open(target_folder + '/' + name + '-invalid.json', 'w') as outfile:
        json.dump(invalid_topo, outfile)
    outfile.close()
    return data[fn]


if __name__ == '__main__':

    json_file = json.load(open("./database/circuit.json"))

    target_folder = './database/'

    invalid_topo = []

    data = {}

    for fn in json_file:

        circuit_a = json_file[fn]['circuit_a'].split('\n')
        circuit_b = json_file[fn]['circuit_b'].split('\n')
        device_list = json_file[fn]['device_list']
        node_list = json_file[fn]['node_list']
        net_list = json_file[fn]['net_list']

        cct_a = Circuit()
        cct_b = Circuit()

        # print(circuit_a)
        # print(circuit_b)

        for item in circuit_a:
            #            if item !='' and item[0]=='V':
            cct_a.add(item)

        for item in circuit_b:
            #            if item !='' and item[0]=='V':
            cct_b.add(item)

        # print(net_list)
        # print(cct_a)
        # print(cct_b)
        # print(device_list)

        #        Vin, Rin, Rout, Rb0, Ra0, Rb1, Ra1, Rb2, Ra2 = symbols('Vin Rin Rout Rb0 Ra0 Rb1 Ra1 Rb2 Ra2')
        #        for item in device_list:
        #               vars()[item]=symbols(item)

        try:
            ss_a = cct_a.ss
            ss_b = cct_b.ss
        except:
            invalid_topo.append(json_file[fn]['key'])
            print("%s violations\n" % fn)
            continue

        a_X = str(ss_a.x)[7:-1]
        a_Y = str(ss_a.y)[7:-1]
        a_A = str(ss_a.A)[7:-1]
        a_B = str(ss_a.B)[7:-1]
        a_C = str(ss_a.C)[7:-1]
        a_D = str(ss_a.D)[7:-1]

        a = {'x': a_X,
             'y': a_Y,
             'a': a_A,
             'b': a_B,
             'c': a_C,
             'd': a_D
             }

        b_X = str(ss_b.x)[7:-1]
        b_Y = str(ss_b.y)[7:-1]
        b_A = str(ss_b.A)[7:-1]
        b_B = str(ss_b.B)[7:-1]
        b_C = str(ss_b.C)[7:-1]
        b_D = str(ss_b.D)[7:-1]

        b = {'x': b_X,
             'y': b_Y,
             'a': b_A,
             'b': b_B,
             'c': b_C,
             'd': b_D
             }

        data[fn] = {'A state': a,
                    'B state': b,
                    'device_list': device_list,
                    'node_list': node_list,
                    'net_list': net_list}

    with open(target_folder + 'expression.json', 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()

    with open(target_folder + 'invalid.json', 'w') as outfile:
        json.dump(invalid_topo, outfile)
    outfile.close()
