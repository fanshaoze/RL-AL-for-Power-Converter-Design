import json
import os

import torch

from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim
from UCT_5_UCB_unblc_restruct_DP_v1.ucts.TopoPlanner import TopoGenState
from UCT_5_UCB_unblc_restruct_DP_v1.ucts.GetReward import calculate_reward

from PM_GNN.code.topo_data import Autopo
from PM_GNN.code.gen_topo_for_dateset import *
from PM_GNN.code.ml_utils import initialize_model


def key_circuit_from_lists(edge_list, node_list, net_list):
    path = find_paths_from_edges(node_list, edge_list)

    node_dic = {}
    node_name = {}
    net_list_dic = {}

    for edge in edge_list:

        edge_start = edge[0]
        edge_end = edge[1]

        if edge_end in node_dic:
            node_dic[edge_end].append(edge_start)
        else:
            node_dic[edge_end] = []
            node_dic[edge_end].append(edge_start)

    for node in node_dic:

        node_dic[node].sort()
        name = 'N'

        for comp in node_dic[node]:
            name = name + '-' + comp

        if node in node_name:
            print('error')
        else:
            node_name[str(node)] = name

    tmp = net_list
    for item in tmp:

        for index, node in enumerate(item):
            if node == '0':
                item[index] = '0'
            elif node in node_name:
                item[index] = node_name[node]
        net_list_dic[item[0]] = item[1::]
        net_list_dic[item[0]].sort()

    net_list_dic_sorted = OrderedDict(sorted(net_list_dic.items()))

    key = str(net_list_dic_sorted)

    return key


def generate_topo_info(topo_data):
    list_of_node = topo_data['list_of_node']
    list_of_edge = topo_data['list_of_edge']
    netlist = topo_data['netlist']
    # TODO has tuple problem
    # key_list = key_circuit_for_single_topo(list_of_edge, list_of_node, netlist)
    # key = key_list[0]
    key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)

    data = {
        "key": key,
        "list_of_node": list_of_node,
        "list_of_edge": list_of_edge,
        "netlist": netlist
    }
    return data


def generate_circuit_info(data):
    net_list = data['netlist']

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

    circuit = {"key": data['key'],
               "circuit_a": circuit_a,
               "circuit_b": circuit_b,
               "device_list": device_list,
               "node_list": node_list,
               "net_list": net_list
               }

    return circuit


def generate_data_for_topo(circuit_topo):
    list_of_node, list_of_edge, netlist, joint_list = \
        convert_to_netlist(circuit_topo.graph, circuit_topo.component_pool, circuit_topo.port_pool,
                           circuit_topo.parent, circuit_topo.comp2port_mapping)
    data = {
        "port_2_idx": circuit_topo.port_2_idx,
        "idx_2_port": circuit_topo.idx_2_port,
        "port_pool": circuit_topo.port_pool,
        "component_pool": circuit_topo.component_pool,
        "same_device_mapping": circuit_topo.same_device_mapping,
        "comp2port_mapping": circuit_topo.comp2port_mapping,
        "port2comp_mapping": circuit_topo.port2comp_mapping,
        "list_of_node": list_of_node,
        "list_of_edge": list_of_edge,
        "netlist": netlist,
        "joint_list": joint_list,
    }
    return data


def assign_DC_C_and_L_in_param(param, fix_paras):
    assert fix_paras['C'] != []
    assert fix_paras['L'] != []
    assert fix_paras['Duty_Cycle'] != []
    param['Duty_Cycle'] = fix_paras['Duty_Cycle']
    param['C'] = fix_paras['C']
    param['L'] = fix_paras['L']
    return param


def generate_topo_for_GNN_model(circuit_topo):
    assert circuit_topo.parameters != []
    fix_paras = {'Duty_Cycle': [circuit_topo.parameters[0]],
                 'C': [circuit_topo.parameters[1]],
                 'L': [circuit_topo.parameters[2]]}
    parameters = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    parameters = assign_DC_C_and_L_in_param(parameters, fix_paras)
    # check
    topo_data = generate_data_for_topo(circuit_topo)
    data_info = generate_topo_info(topo_data)
    circuit_info = generate_circuit_info(data_info)  # cki

    # with open('./database/analytic.csv', newline='') as f:
    # reader = csv.reader(f)
    # result_analytic = list(reader)

    dataset = {}

    # n_os = 100

    device_list = circuit_info["device_list"]
    num_dev = len(device_list) - 3
    param2sweep, paramname = gen_param(device_list, parameters)
    paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

    name_list = {}
    for index, name in enumerate(paramname):
        name_list[name] = index

    count = 0
    tmp_device_name = ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"] + device_list[-num_dev:]

    device_name = {}

    for i, item in enumerate(tmp_device_name):
        device_name[item] = i

    count = 0
    for vect in paramall:
        edge_attr = {}
        edge_attr0 = {}
        node_attr = {}
        node_attr["VIN"] = [1, 0, 0, 0]
        node_attr["VOUT"] = [0, 1, 0, 0]
        node_attr["GND"] = [0, 0, 1, 0]

        for val, key in enumerate(device_name):
            if key in ["Duty_Cycle", "Frequency", "Vin", "Rout", "Cout", "Rin"]:
                continue
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

        for item in data_info["list_of_node"]:
            if str(item).isnumeric():
                node_attr[str(item)] = [0, 0, 0, 1]
        dataset[str(count)] = {"list_of_edge": data_info["list_of_edge"],
                               "list_of_node": data_info["list_of_node"],
                               "netlist": data_info["netlist"],
                               "edge_attr": edge_attr,
                               "edge_attr0": edge_attr0,
                               "node_attr": node_attr,
                               "duty_cycle": vect[device_name["Duty_Cycle"]],
                               "rout": vect[device_name["Rout"]],
                               "cout": vect[device_name["Cout"]],
                               "freq": vect[device_name["Frequency"]]
                               }
        count = count + 1
    print('dataset: ', dataset)

    return dataset


class GNNRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_model_file, vout_model_file, reward_model_file, debug=False, *args):
        super().__init__(debug, *args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # The round here means that we round the vout to the range[0,1],
        # and directly use the round(vout)*efficiency as the reward
        if self.configs_['reward_model'] is not None:

            self.reward_model = self.load_model(self.reward_y_select)
        else:
            self.reward_y_select = 'None'
            self.eff_y_select = eff_model_file
            self.vout_y_select = vout_model_file

            self.eff_model = self.load_model(self.eff_y_select)
            self.vout_model = self.load_model(self.vout_y_select)
        self.raw_dataset_file = 'raw_dataset.json'
        self.reg_data_folder = './PM_GNN/2_dataset/'

        #
        self.num_node = self.configs_['nnode']
        self.batch_size = 1

    def load_model(self, y_select, output_size=None):
        # device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = initialize_model(model_index=1, gnn_nodes=100, pred_nodes=100, gnn_layers=3,
        #                          nf_size=4, ef_size=3, device=device_)
        # model = initialize_model(model_index=1, gnn_nodes=100, pred_nodes=100, gnn_layers=3,
        #                          nf_size=4, ef_size=3, device=self.device)
        # model = initialize_model(model_index=1, gnn_nodes=100, pred_nodes=100, gnn_layers=4,
        #                          nf_size=4, ef_size=3, device=self.device) # for pre reg_eff/vout.pt
        nf_size = 4
        ef_size = 3
        nnode = 7

        model = initialize_model(model_index=self.configs_['model_index'],
                                 gnn_nodes=self.configs_['gnn_nodes'],
                                 pred_nodes=self.configs_['predictor_nodes'],
                                 gnn_layers=self.configs_['gnn_layers'],
                                 nf_size=nf_size, ef_size=ef_size, device=self.device)
        # assign the pt file name, the select is the file name without type
        pt_filename = y_select + '.pt'
        if os.path.exists(pt_filename):
            print('loading model from pt file', y_select)
            model_state_dict, _ = torch.load(pt_filename)
            model.load_state_dict(model_state_dict)
        return model

    def get_output_with_model(self, data, effi_model, model_index):
        effi_model.eval()
        accuracy = 0
        n_batch_test = 0
        data.to(self.device)
        L = data.node_attr.shape[0]
        B = int(L / self.num_node)
        node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
        if model_index == 0:
            edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
        else:
            edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
            edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

        adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])
        # y = data.label.cpu().detach().numpy()

        n_batch_test = n_batch_test + 1
        if model_index == 0:
            out = effi_model(input=(node_attr.to(self.device),
                                    edge_attr.to(self.device),
                                    adj.to(self.device), self.configs_['gnn_layers'])).cpu().detach().numpy()
        else:
            out = effi_model(input=(node_attr.to(self.device),
                                    edge_attr1.to(self.device),
                                    edge_attr2.to(self.device),
                                    adj.to(self.device), self.configs_['gnn_layers'])).cpu().detach().numpy()
        return out

    def get_surrogate_output_using_rawdata(self, effi_or_vout):
        if (effi_or_vout == 'effi') or (effi_or_vout == 'eff'):
            out_effi_list = []
            dataset = Autopo(self.raw_dataset_file, self.reg_data_folder + self.eff_y_select, self.eff_y_select)
            print(len(dataset))
            # _, _, test_loader = split_balance_data(dataset, False, batch_size=self.batch_size)
            # print(len(test_loader))
            test_loader = dataset
            for data in test_loader:
                out_effi = self.get_output_with_model(data, self.eff_model, model_index='model_index')
                out_effi_list.append(out_effi)
            return out_effi_list[0][0][0]
        elif effi_or_vout == 'vout':
            out_vout_list = []
            dataset = Autopo(self.raw_dataset_file, self.reg_data_folder + self.vout_y_select, self.vout_y_select)
            # _, _, test_loader = split_balance_data(dataset, False, batch_size=self.batch_size)
            test_loader = dataset
            for data in test_loader:
                out_vout = self.get_output_with_model(data, self.vout_model, model_index='model_index')
                out_vout_list.append(out_vout)
            return 100 * out_vout_list[0][0][0]
        elif effi_or_vout == 'reward':
            out_reward_list = []
            dataset = Autopo(self.raw_dataset_file, self.reg_data_folder + self.reward_y_select, self.reward_y_select)
            print(len(dataset))
            # _, _, test_loader = split_balance_data(dataset, False, batch_size=self.batch_size)
            # print(len(test_loader))
            test_loader = dataset
            for data in test_loader:
                out_reward = self.get_output_with_model(data, self.reward_model, model_index='model_index')
                out_reward_list.append(out_reward)
            return out_reward_list[0][0][0]

    def get_surrogate_reward(self, state: TopoGenState):
        if state.parameters == []:  # eff, vout, reward, parameter
            return 0, -500, 0, []
        os.system('rm ' + self.reg_data_folder + self.eff_y_select + '/processed/data.pt')
        os.system('rm ' + self.reg_data_folder + self.vout_y_select + '/processed/data.pt')
        os.system('rm ' + self.reg_data_folder + self.reward_y_select + '/processed/data.pt')
        # TODO update
        raw_dataset = generate_topo_for_GNN_model(state)
        # save raw dataset in self.reg_data_folder + self.eff_y_select + '/' + self.raw_dataset_file
        self.save_dataset_to_file(raw_dataset)

        print(len(raw_dataset))
        # get_effciencies
        # dataset = Autopo(self.raw_dataset_file, self.reg_data_folder + self.eff_y_select, self.eff_y_select)
        # print(len(dataset))
        # # _, _, test_loader = split_balance_data(dataset, False, batch_size=self.batch_size)
        # # print(len(test_loader))
        # test_loader = dataset
        # for data in test_loader:
        #     out_effi = get_output_with_model(data, self.eff_model, self.device,
        #                                      num_node=self.num_node, model_index=1)
        #     out_effi_list.append(out_effi)

        # dataset = Autopo(self.raw_dataset_file, self.reg_data_folder + self.vout_y_select, self.vout_y_select)
        # # _, _, test_loader = split_balance_data(dataset, False, batch_size=self.batch_size)
        # test_loader = dataset
        # for data in test_loader:
        #     out_vout = get_output_with_model(data, self.vout_model, self.device,
        #                                      num_node=self.num_node, model_index=1)
        #     out_vout_list.append(out_vout)
        # print(out_effi_list, out_vout_list)
        # eff = out_effi_list[0][0][0]
        # vout = 100 * out_vout_list[0][0][0]
        start_time = time.time()

        if self.configs_['reward_model'] is not None:
            reward = self.get_surrogate_output_using_rawdata(effi_or_vout='reward')
            # if directly predict reward and not know eff and vout,
            # we set eff as reward and vout as target vout
            eff, vout = reward, self.configs_['target_vout']
        else:
            eff = self.get_surrogate_output_using_rawdata(effi_or_vout='eff')
            vout = self.get_surrogate_output_using_rawdata(effi_or_vout='vout')

            eff_obj = {'efficiency': float(eff), 'output_voltage': float(vout)}
            if self.configs_['round'] == 'vout':
                reward = float(eff) * float(vout) / 100
            else:
                reward = calculate_reward(eff_obj, self.configs_['target_vout'])

        self.new_query_time += time.time() - start_time
        self.new_query_counter += 1

        os.system('rm ' + self.reg_data_folder + self.eff_y_select + '/processed/data.pt')
        os.system('rm ' + self.reg_data_folder + self.vout_y_select + '/processed/data.pt')
        os.system('rm ' + self.reg_data_folder + self.reward_y_select + '/processed/data.pt')

        # get max reward

        return eff, vout, reward, state.parameters

    def save_dataset_to_file(self, dataset):
        if self.configs_['reward_model'] is not None:
            with open(self.reg_data_folder + self.reward_y_select + '/' + self.raw_dataset_file, 'w') as f:
                json.dump(dataset, f)
            f.close()
        else:
            with open(self.reg_data_folder + self.eff_y_select + '/' + self.raw_dataset_file, 'w') as f:
                json.dump(dataset, f)
            f.close()
            with open(self.reg_data_folder + self.vout_y_select + '/' + self.raw_dataset_file, 'w') as f:
                json.dump(dataset, f)
            f.close()

    def get_surrogate_vout(self, state: TopoGenState):
        # TODO
        pass

    def get_surrogate_eff(self, state: TopoGenState):
        """
        return the eff prediction of state, and of self.get_state() if None
        """
        pass

    def get_single_topo_sim_result(self, state: TopoGenState):
        pass
