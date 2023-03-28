import warnings

from SimulatorAnalysis.gen_topo import *
from SimulatorAnalysis.random_topo import generate_one_data_json
from SimulatorAnalysis.data import generate_key_data
from SimulatorAnalysis.circuit_val import get_one_circuit
from SimulatorAnalysis.expr_val import get_one_expression
from SimulatorAnalysis.analytics import get_analytics_result, get_analytics_information

anal_root_folder = './UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis'
external_data_base_folder = anal_root_folder + '/database'


def generate_circuit_info_in_target_folder(target_folder, file_name, graph, comp2port_mapping, port2comp_mapping,
                                           idx_2_port, port_2_idx,
                                           parent, component_pool, same_device_mapping, port_pool, fix_paras,
                                           pass_para=True):
    directory_path = anal_root_folder + '/components_data_random'
    return_info_data_json = generate_one_data_json(file_name, directory_path, graph, comp2port_mapping,
                                                   port2comp_mapping,
                                                   idx_2_port, port_2_idx, parent,
                                                   component_pool, same_device_mapping, port_pool)

    if return_info_data_json == "Has prohibit path" or \
            return_info_data_json == "Not has switch" or \
            return_info_data_json == "Has redundant loop":
        return None, None
    key = generate_key_data(directory_path, file_name, target_folder)
    circuit = get_one_circuit(target_folder=target_folder, name=file_name, fix_paras=fix_paras, pass_para=pass_para)
    return key, circuit['device_list']


# TODO funtion this need to be updated for the new key expression format
def key_expression_dict(target_vout_min=-500, target_vout_max=500):
    data_json_file = json.load(open(external_data_base_folder + '/data.json'))
    # data_json_file = json.load(open("./database/data.json"))
    # print(len(data_json_file))
    expression_json_file = json.load(
        open(external_data_base_folder + '/expression.json'))
    # expression_json_file = json.load(open("./database/expression.json"))

    # print("length of expression_json_file:", len(expression_json_file))
    simu_results = []
    with open(external_data_base_folder + '/analytic.csv', 'r') as csv_file:
        # with open("./database/analytic.csv", 'r') as csv_file:

        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            simu_results.append(row)

    print(len(simu_results))
    print(len(data_json_file))
    print(len(expression_json_file))

    data = {}
    key_expression = {}
    x = 0
    for data_fn in data_json_file:
        # print(x)
        if data_fn in expression_json_file:
            expression = expression_json_file[data_fn]
        else:
            expression = "Invalid"
            duty_cycle_para = -1
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para] = {}
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para]['Expression'] = expression
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para]['Efficiency'] = 0
            key_expression[data_json_file[data_fn]['key'] + '$' + duty_cycle_para]['Vout'] = target_vout_min

        for i in range(len(simu_results)):
            if simu_results[i][0] == data_fn:

                while i < (len(simu_results)) and simu_results[i][0] == data_fn:
                    duty_cycle_para = float(simu_results[i][1])
                    # print(duty_cycle_para)
                    assert not (data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para) in key_expression)

                    # print(data_json_file[data_fn]['key'] + '$' + paras_str)
                    key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)] = {}
                    key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)][
                        'Expression'] = expression
                    if simu_results[i][3] != 'False' and simu_results[i][4] != 'False':

                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Efficiency'] = \
                            simu_results[i][4]
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Vout'] = \
                            simu_results[i][3]
                    else:
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Efficiency'] = 0
                        key_expression[data_json_file[data_fn]['key'] + '$' + str(duty_cycle_para)]['Vout'] = \
                            target_vout_min
                    i += 1
                break
        x += 1
    # with open('./database/analytic-expression.json', 'w') as f:
    with open(external_data_base_folder + '/analytic-expression.json', 'w') as f:
        json.dump(key_expression, f)
    f.close()

    return key_expression


def save_one_analytics_result(key_expression):
    pass


def generate_para_strs(device_list, duty_cycle):
    para_strs = []
    parameters = json.load(open(anal_root_folder + '/param.json'))
    if duty_cycle != -1:
        parameters['Duty_Cycle'] = [duty_cycle]
    else:
        return None
    param2sweep, paramname = gen_param(device_list, parameters)
    # print(param2sweep)
    # print(paramname)
    paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))
    # print(paramall)
    name_list = {}
    for index, name in enumerate(paramname):
        name_list[name] = index

    for vect in paramall:
        para_strs.append(str(vect))

    return para_strs


def find_one_analytics_result(current, key, key_expression, graph, comp2port_mapping, port2comp_mapping, idx_2_port,
                              port_2_idx,
                              parent,
                              component_pool, same_device_mapping, port_pool, duty_cycle=-1):
    """
    We use the key information to get one analytics' result
    :param key: uniquely representing a topology
    :param key_expression: key expression mapping
    :param graph:
    :param comp2port_mapping:
    :param port2comp_mapping:
    :param idx_2_port:
    :param port_2_idx:
    :param parent:
    :param component_pool:
    :param same_device_mapping:
    :param port_pool:
    :return: analytics result
    """
    para_result = {}
    if key + str(current.parameters) in key_expression:
        return {str(current.parameters): [key_expression[key + '$' + str(current.parameters)]['Effiency'],
                                          key_expression[key + '$' + str(current.parameters)]['Vout']]}
    else:
        find_flag = 0
        # device_list = get_one_device_list(graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx, parent,
        #                                   component_pool, same_device_mapping, port_pool)
        key, device_list = generate_circuit_info_in_target_folder(
            external_data_base_folder + '',
            "PCC-device-list", graph, comp2port_mapping,
            port2comp_mapping,
            idx_2_port, port_2_idx, parent, component_pool,
            same_device_mapping, port_pool)
        if not device_list:
            return None
        paras_str = generate_para_strs(device_list, duty_cycle)
        print(paras_str)

        for para_str in paras_str:
            paras = para_str[1:-2].split(',')
            duty_cycle_para = float(paras[0])
            # duty_cycle_para = float(simu_results[i][1])

            if key + '$' + str(duty_cycle_para) in key_expression:
                find_flag = 1
                para_result[str(duty_cycle_para)] = [key_expression[key + '$' + str(duty_cycle_para)]['Efficiency'],
                                                     key_expression[key + '$' + str(duty_cycle_para)]['Vout']]

    if find_flag == 1:
        return para_result
    return None


# def get_one_device_list(graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx, parent, component_pool,
#                         same_device_mapping, port_pool):
#     return generate_circuit_info_in_target_folder("PCC-device-list", graph, comp2port_mapping, port2comp_mapping,
#                                                   idx_2_port, port_2_idx, parent, component_pool,
#                                                   same_device_mapping, port_pool)
# directory_path = anal_root_folder + '/components_data_random'
# target_folder = anal_files_base_folder + ''
# return_info_data_json = generate_one_data_json(name, directory_path, graph, comp2port_mapping, port2comp_mapping,
#                                                idx_2_port, port_2_idx, parent,
#                                                component_pool, same_device_mapping, port_pool)
# if return_info_data_json == "Has prohibit path" or \
#         return_info_data_json == "Not has switch" or \
#         return_info_data_json == "Has redundant loop":
#     return None
# key = generate_key_data(directory_path, name, target_folder)
# circuit = get_one_circuit(target_folder, name)
# return circuit['device_list']


def get_analytics_file(key_expression, graph, comp2port_mapping, port2comp_mapping, idx_2_port, port_2_idx,
                       parent, component_pool,
                       same_device_mapping, port_pool, fix_paras, target_vout_min=-500):
    para_result = {}
    pid_label = str(os.getpid())
    name = "PCC-" + pid_label
    target_folder = external_data_base_folder + ''
    key, device_list = generate_circuit_info_in_target_folder(target_folder, name, graph, comp2port_mapping,
                                                              port2comp_mapping,
                                                              idx_2_port, port_2_idx, parent, component_pool,
                                                              same_device_mapping, port_pool, fix_paras)
    # directory_path = anal_root_folder + '/components_data_random'
    # 
    # return_info_data_json = generate_one_data_json(name, directory_path, graph, comp2port_mapping, port2comp_mapping,
    #                                                idx_2_port, port_2_idx, parent,
    #                                                component_pool, same_device_mapping, port_pool)
    # if return_info_data_json == "Has prohibit path" or \
    #         return_info_data_json == "Not has switch" or \
    #         return_info_data_json == "Has redundant loop":
    #     return None
    # key = generate_key_data(directory_path, name, target_folder)
    # circuit = get_one_circuit(target_folder, name)
    if key is None and device_list is None:
        # means 'has redundant loop' or some other
        return None
    # key_expression:   key+$+'[C,L]':{'Expression': expression,'duty cycle':{'Efficiency':efficiency, 'Vout':vout}}
    para_list = [fix_paras['Duty_Cycle'][0], fix_paras['C'][0], fix_paras['L'][0]]
    C_L_list_str = str([para_list[1], para_list[2]])
    if key + '$' + C_L_list_str in key_expression:
        expression = key_expression[key + '$' + C_L_list_str]['Expression']
    else:
        expression = get_one_expression(target_folder, name)
    anal_info = None
    if str(expression) == "invalid":
        if key + '$' + C_L_list_str not in key_expression:
            key_expression[key + '$' + C_L_list_str] = {}
        key_expression[key + '$' + C_L_list_str]['Expression'] = expression
        key_expression[key + '$' + C_L_list_str][str(para_list[0])] = {}
        key_expression[key + '$' + C_L_list_str][str(para_list[0])]['Efficiency'] = 0
        key_expression[key + '$' + C_L_list_str][str(para_list[0])]['Vout'] = target_vout_min
        return None
    else:
        # para is None for try all paras
        anal_info = get_analytics_information(target_folder, name, fix_paras=fix_paras, expression=expression)
        # anal_info: [name_list, net_list, vout, efficiency, flag_candidate, 1]
        for fn, info in anal_info.items():
            for k, info_list in info.items():
                if key + '$' + C_L_list_str not in key_expression:
                    key_expression[key + '$' + C_L_list_str] = {}
                key_expression[key + '$' + C_L_list_str]['Expression'] = expression
                key_expression[key + '$' + C_L_list_str][str(para_list[0])] = {}
                if info_list[2] != 'False' and info_list[3] != 'False':
                    key_expression[key + '$' + C_L_list_str][str(para_list[0])]['Efficiency'] = \
                        info_list[3]
                    key_expression[key + '$' + C_L_list_str][str(para_list[0])]['Vout'] = \
                        info_list[2]
                else:
                    key_expression[key + '$' + C_L_list_str][str(para_list[0])]['Efficiency'] = 0
                    key_expression[key + '$' + C_L_list_str][str(para_list[0])][
                        'Vout'] = target_vout_min
    print(anal_info)
    return anal_info


def simulate_one_analytics_result(analytics_info):
    if analytics_info == None:
        return {'result_valid': False, 'efficiency': 0, 'Vout': 0}
    analytic = analytics_info
    for fn in analytic:
        print(int(fn[6:9]))
        count = 'tmp_simu'
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = fn + '-' + str(count) + '.cki'
            convert_cki_full_path(file_name, param_value, device_name, netlist)
            print(file_name)
            simulate(file_name)
    data = {}
    data_csv = []
    for fn in analytic:
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = fn + '-' + str(count) + '.simu'
            path = file_name
            vin = param_value[device_name['Vin']]
            freq = param_value[device_name['Frequency']] * 1000000
            rin = param_value[device_name['Rin']]
            rout = param_value[device_name['Rout']]
            print(file_name)
            result = calculate_efficiency(path, vin, freq, rin, rout)
            print(result)
            return result


def get_one_analytics_result(current, key_expression, graph, comp2port_mapping, port2comp_mapping, idx_2_port,
                             port_2_idx, parent, component_pool, same_device_mapping, port_pool,
                             using_hash=True, target_vout_min=-500):
    """

    @param current:
    @param key_expression:
    @param graph:
    @param comp2port_mapping:
    @param port2comp_mapping:
    @param idx_2_port:
    @param port_2_idx:
    @param parent:
    @param component_pool:
    @param same_device_mapping:
    @param port_pool:
    @param using_hash:
    @param target_vout_min:
    @return:
    """
    para_result, pid_label, name, target_folder = {}, str(os.getpid()), "PCC-" + str(
        os.getpid()), external_data_base_folder + ''
    key, device_list = generate_circuit_info_in_target_folder(target_folder=target_folder, file_name=name, graph=graph,
                                                              comp2port_mapping=comp2port_mapping,
                                                              port2comp_mapping=port2comp_mapping,
                                                              idx_2_port=idx_2_port, port_2_idx=port_2_idx,
                                                              parent=parent,
                                                              component_pool=component_pool,
                                                              same_device_mapping=same_device_mapping,
                                                              port_pool=port_pool,
                                                              fix_paras={'Duty_Cycle': [current.parameters[0]],
                                                                         'C': [current.parameters[1]],
                                                                         'L': [current.parameters[2]]},
                                                              pass_para=True)
    if not device_list:
        return None, 0

    C_L_list_str, duty_str = str([current.parameters[1], current.parameters[2]]), str(current.parameters[0])

    # get expression
    # key_expression: key+$+'[C,L]':{'Expression': expression,'duty cycle':{'Efficiency':efficiency, 'Vout':vout}}
    if using_hash and key + '$' + C_L_list_str in key_expression:
        expression, expression_time = key_expression[key + '$' + C_L_list_str]['Expression'], 0
    else:
        start_time = time.time()
        expression = get_one_expression(target_folder, name)
        print(f'-------------getting expression using: {time.time() - start_time} sec.')
        expression_time = time.time() - start_time

    # get result using expression
    if str(expression) == "invalid":
        if key + '$' + C_L_list_str not in key_expression:
            key_expression[key + '$' + C_L_list_str] = {'Expression': expression,
                                                        duty_str: {'Efficiency': 0, 'Vout': target_vout_min}}
        return None, expression_time
    else:
        # get anal_results in format [fn, duty_cycle, str(vect), vout, efficiency, flag_candidate] with len=1
        anal_results = get_analytics_result(target_folder=target_folder, name=name,
                                            fix_paras={'Duty_Cycle': [current.parameters[0]],
                                                       'C': [current.parameters[1]],
                                                       'L': [current.parameters[2]]}, expression=expression)
        assert len(anal_results) == 1
        anal_result = anal_results[0]
        # update the key_expression
        if key + '$' + C_L_list_str not in key_expression:
            key_expression[key + '$' + C_L_list_str] = {'Expression': expression}
        if duty_str not in key_expression[key + '$' + C_L_list_str]:
            if anal_result[3] != 'False' and anal_result[4] != 'False':
                key_expression[key + '$' + C_L_list_str][duty_str] = {'Efficiency': anal_result[4],
                                                                      'Vout': anal_result[3]}
            else:
                if (not anal_result[3]) and (type(anal_result[3]) is bool):  # tmp-check the type of the anal results
                    warnings.warn('error of the output types of eff and vout')
                key_expression[key + '$' + C_L_list_str][duty_str] = {'Efficiency': 0, 'Vout': target_vout_min}
        para_result[str(current.parameters)] = [key_expression[key + '$' + C_L_list_str][duty_str]['Efficiency'],
                                                key_expression[key + '$' + C_L_list_str][duty_str]['Vout']]
        return para_result, expression_time


def read_analytics_result():
    data_json_file = json.load(
        open(external_data_base_folder + '/analytic-expression.json'))
    return data_json_file


def read_no_sweep_analytics_result():
    data_json_file = json.load(
        open(external_data_base_folder + '/no-sweep-analytic-expression.json'))

    return data_json_file


def read_no_sweep_sim_result():
    data_json_file = json.load(
        open(external_data_base_folder + '/no-sweep-key-sim-result.json'))
    return data_json_file


def save_no_sweep_analytics_result(anal_expressions):
    data_json_file = json.load(
        open(external_data_base_folder + '/no-sweep-analytic-expression.json'))
    data_json_file.update(anal_expressions)
    with open(external_data_base_folder + '/no-sweep-analytic-expression.json',
              'w') as f:
        json.dump(data_json_file, f)
    f.close()


def save_no_sweep_sim_result(simu_results):
    data_json_file = json.load(
        open(external_data_base_folder + '/no-sweep-key-sim-result.json.json'))
    data_json_file.update(simu_results)
    with open(external_data_base_folder + '/no-sweep-key-sim-result.json', 'w') as f:
        json.dump(data_json_file, f)
    f.close()


def save_sta_result(simu_results, file_name):
    with open(file_name, 'w') as f:
        json.dump(simu_results, f)
    f.close()


if __name__ == '__main__':
    key_expression_dict()
