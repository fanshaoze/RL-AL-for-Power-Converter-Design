from GNN_gendata.gen_topo import *


def assign_DC_C_and_L_in_param(param, fix_paras):
    assert fix_paras['C'] != []
    assert fix_paras['L'] != []
    assert fix_paras['Duty_Cycle'] != []
    param['Duty_Cycle'] = fix_paras['Duty_Cycle']
    param['C'] = fix_paras['C']
    param['L'] = fix_paras['L']
    return param


def get_dataset(fix_paras, init_data, key_data, circuit_data, eff, vout, eff_analytic=0, vout_analytic=0):
    parameters = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    parameters = assign_DC_C_and_L_in_param(parameters, fix_paras)

    dataset = {}

    n_os = 100

    # print(fn)
    # fn_data = fn
    #
    # print(fn_data)

    device_list = circuit_data["device_list"]
    num_dev = len(device_list) - 3
    param2sweep, paramname = gen_param(device_list, parameters)
    paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

    name_list = {}
    for index, name in enumerate(paramname):
        name_list[name] = index

    tmp_device_name = ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"] + device_list[-num_dev:]

    device_name = {}

    for i, item in enumerate(tmp_device_name):
        device_name[item] = i

    vect = paramall[0]
    edge_attr = {}
    edge_attr0 = {}
    node_attr = {}
    node_attr["VIN"] = [1, 0, 0, 0]
    node_attr["VOUT"] = [0, 1, 0, 0]
    node_attr["GND"] = [0, 0, 1, 0]
    for val, key in enumerate(device_name):
        if key in ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"]:
            continue
        #                   print(key)
        duty_cycle = vect[device_name["Duty_Cycle"]]
        if key[:2] == 'Ra':
            edge_attr['Sa' + key[2]] = [1 / float(vect[val]) * duty_cycle, 0, 0]
            edge_attr0['Sa' + key[2]] = [float(vect[val]), 0, 0, 0, 0, duty_cycle]
        elif key[:2] == 'Rb':
            edge_attr['Sb' + key[2]] = [1 / float(vect[val]) * (1 - duty_cycle), 0, 0]
            edge_attr0['Sb' + key[2]] = [0, float(vect[val]), 0, 0, 0, duty_cycle]
        elif key[0] == 'C':
            edge_attr[key] = [0, float(vect[val]), 0]
            edge_attr0[key] = [0, 0, vect[val], 0, 0, 0]
        elif key[0] == 'L':
            edge_attr[key] = [0, 0, 1 / float(vect[val])]
            edge_attr0[key] = [0, 0, 0, vect[val], 0, 0]
        else:
            edge_attr[key] = [0, 0, 0, 0, 0, 0]

        #               print(edge_attr)

    for item in key_data["list_of_node"]:
        if str(item).isnumeric():
            node_attr[str(item)] = [0, 0, 0, 1]

    one_dataset = {"list_of_edge": key_data["list_of_edge"],
                   "list_of_node": key_data["list_of_node"],
                   "netlist": key_data["netlist"],
                   "edge_attr": edge_attr,
                   "edge_attr0": edge_attr0,
                   "node_attr": node_attr,
                   "vout": vout,
                   "eff": eff,
                   "vout_analytic": vout_analytic,
                   "eff_analytic": eff_analytic,
                   "duty_cycle": vect[device_name["Duty_Cycle"]],
                   "rout": vect[device_name["Rout"]],
                   "cout": vect[device_name["Cout"]],
                   "freq": vect[device_name["Frequency"]]
                   }
    return one_dataset


if __name__ == '__main__':

    cki = json.load(open("database/circuit.json"))
    parameters = json.load(open("param.json"))

    data = json.load(open("database/data.json"))
    result = json.load(open("parallel/sim.json"))

    #       with open('./database/analytic.csv', newline='') as f:
    #           reader = csv.reader(f)
    #           result_analytic = list(reader)

    dataset = {}

    n_os = 100

    for fn in cki:

        print(fn)
        fn_data = fn

        #           if len(fn)==9:
        #               fn_data=fn[0:5]+'00'+fn[-4:]
        #           if len(fn)==10:
        #               fn_data=fn[0:5]+'0'+fn[-5:]

        print(fn_data)

        device_list = cki[fn]["device_list"]
        num_dev = len(device_list) - 3
        param2sweep, paramname = gen_param(device_list, parameters)
        paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

        #           print(paramall)
        #           print(device_list)
        #           print(param2sweep)
        #           print(paramname)
        #           print(fn, paramall)
        name_list = {}
        for index, name in enumerate(paramname):
            name_list[name] = index

        count = 0
        tmp_device_name = ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"] + device_list[-num_dev:]

        device_name = {}

        for i, item in enumerate(tmp_device_name):
            device_name[item] = i

        count = 0

        tmp_list_analytic = []

        for vect in paramall:
            edge_attr = {}
            edge_attr0 = {}
            node_attr = {}
            node_attr["VIN"] = [1, 0, 0, 0]
            node_attr["VOUT"] = [0, 1, 0, 0]
            node_attr["GND"] = [0, 0, 1, 0]

            #               print(device_name)

            for val, key in enumerate(device_name):
                if key in ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"]:
                    continue
                #                   print(key)
                duty_cycle = vect[device_name["Duty_Cycle"]]
                if key[:2] == 'Ra':
                    edge_attr['Sa' + key[2]] = [1 / float(vect[val]) * duty_cycle, 0, 0]
                    edge_attr0['Sa' + key[2]] = [float(vect[val]), 0, 0, 0, 0, duty_cycle]
                elif key[:2] == 'Rb':
                    edge_attr['Sb' + key[2]] = [1 / float(vect[val]) * (1 - duty_cycle), 0, 0]
                    edge_attr0['Sb' + key[2]] = [0, float(vect[val]), 0, 0, 0, duty_cycle]
                elif key[0] == 'C':
                    edge_attr[key] = [0, float(vect[val]), 0]
                    edge_attr0[key] = [0, 0, vect[val], 0, 0, 0]
                elif key[0] == 'L':
                    edge_attr[key] = [0, 0, 1 / float(vect[val])]
                    edge_attr0[key] = [0, 0, 0, vect[val], 0, 0]
                else:
                    edge_attr[key] = [0, 0, 0, 0, 0, 0]

            #               print(edge_attr)

            for item in data[fn_data]["list_of_node"]:
                if str(item).isnumeric():
                    node_attr[str(item)] = [0, 0, 0, 1]

            #               print(vect)

            try:
                name_key = str(vect[0]) + '-' + str(vect[3])
                vout = result[fn][name_key]["vout"]
                eff = result[fn][name_key]["eff"]
                print(name_key, 'here')
            except:
                continue

            flag = 0
            if flag == 0:
                vout_analytic = 0
                eff_analytic = 0

            #               print(data[fn])

            dataset[fn + '-' + str(count)] = {"list_of_edge": data[fn]["list_of_edge"],
                                              "list_of_node": data[fn]["list_of_node"],
                                              "netlist": data[fn]["netlist"],
                                              "edge_attr": edge_attr,
                                              "edge_attr0": edge_attr0,
                                              "node_attr": node_attr,
                                              "vout": vout,
                                              "eff": eff,
                                              "vout_analytic": vout_analytic,
                                              "eff_analytic": eff_analytic,
                                              "duty_cycle": vect[device_name["Duty_Cycle"]],
                                              "rout": vect[device_name["Rout"]],
                                              "cout": vect[device_name["Cout"]],
                                              "freq": vect[device_name["Frequency"]]

                                              }

            #               print(dataset[fn+'-'+str(count)])
            #               if abs(vout)>110:
            #                   count_os=0
            #                   for sss in range(0):
            #                      dataset[fn+'-'+str(count)+'-boost-'+str(count_os)]={"list_of_edge": data[fn]["list_of_edge"],
            #                            "list_of_node": data[fn]["list_of_node"],
            #                            "netlist": data[fn]["netlist"],
            #                            "edge_attr": edge_attr,
            #                            "node_attr": node_attr,
            #                            "vout": vout,
            #                            "eff": eff,
            #                            "duty_cycle":vect[device_name["Duty_Cycle"]],
            #                            "rout":vect[device_name["Rout"]],
            #                            "cout":vect[device_name["Cout"]],
            #                            "freq":vect[device_name["Frequency"]]
            #                        }
            #                      count_os=count_os+1
            #
            #               if 30<abs(vout)<70:
            #                   count_os=0
            #                   for sss in range(0):
            #                      dataset[fn+'-'+str(count)+'-buck-'+str(count_os)]={"list_of_edge": data[fn]["list_of_edge"],
            #                            "list_of_node": data[fn]["list_of_node"],
            #                            "netlist": data[fn]["netlist"],
            #                            "edge_attr": edge_attr,
            #                            "node_attr": node_attr,
            #                            "vout": vout,
            #                            "eff": eff,
            #                            "duty_cycle":vect[device_name["Duty_Cycle"]],
            #                            "rout":vect[device_name["Rout"]],
            #                            "cout":vect[device_name["Cout"]],
            #                            "freq":vect[device_name["Frequency"]]
            #                        }
            #                      count_os=count_os+1

            count = count + 1

    with open('./database/dataset.json', 'w') as f:
        json.dump(dataset, f)
    f.close()
