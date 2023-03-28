import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from collections import defaultdict
from shutil import copyfile
import os
import subprocess
import argparse
import numpy as np
import json

def calculate_efficiency(path, input_voltage, freq):
    # my_timeout = 5
    # info = 0  # this means no error occurred
    simu_file = path[:-3] + 'simu'
    file = open(simu_file, 'r')
    V_in = input_voltage
    V_out = []
    I_in = []
    I_out = []
    time = []
    stable_ratio = 0.01

    cycle = 1 / freq
    # count = 0


    read_V_out, read_I_out, read_I_in = False, False, False
    for line in file:
        # print(line)
        if "Transient solution failed" in line:
            return {'result_valid': False,
                    'efficiency': 0.0,
                    'error_msg': 'transient_simulation_failure'}
        if "Index   time            v(out)" in line and not read_V_out:
            read_V_out = True
            read_I_out = False
            read_I_in = False
            continue
        elif "Index   time            v(in_ext,in)" in line and not read_I_in:
            read_V_out = False
            read_I_out = False
            read_I_in = True
            continue
        elif "Index   time            v(out,out_ext)" in line and not read_I_out:
            read_V_out = False
            read_I_out = True
            read_I_in = False
            continue

        tokens = line.split()

        # print(tokens)
        if len(tokens) == 3 and tokens[0] != "Index":
            if read_V_out:
                time.append(float(tokens[1]))
                try:
                    V_out.append(float(tokens[2]))
                except:
                    V_out.append(0)
            elif read_I_in:
                try:
                   I_in.append(float(tokens[2])/0.001)
                except:
                    I_in.append(0)
            elif read_I_out:
                try:
                  I_out.append(float(tokens[2])/0.001)
                except:
                  I_out.append(0)

    #print(len(V_out),len(I_out),len(I_in),len(time))
    if len(V_out) == len(I_out) == len(I_in) == len(time):
        pass
    else:
        print("don't match")
        return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'output_is_not_aligned'}

    if not V_out or not I_in or not I_out:
        print(V_out)
        return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'missing_output_type'}

    # print(I_out, I_in)
    end = len(V_out) - 1
    start = len(V_out) - 1
    while start >= 0:
        if time[end] - time[start] >= 50 * cycle:
            break
        start -= 1

    if start == -1:
        print("duration less than one cycle")
        return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'less_than_one_cycle'}
    mid = int((start + end)/2)

    # print(start, end,time[end] - time[start])
    P_in = sum([(I_in[x] + I_in[x + 1]) / 2 * (V_in + V_in) / 2 *
                (time[x + 1] - time[x])
                for x in range(start, end)]) / (time[end] - time[start])

    P_out = sum([(I_out[x] + I_out[x + 1]) / 2 * (V_out[x] + V_out[x + 1]) / 2 *
                 (time[x + 1] - time[x])
                 for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave = sum([(V_out[x] + V_out[x + 1]) / 2 * (time[x + 1] - time[x])
                 for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave_1 = np.average(V_out[start:mid])

    V_out_ave_2 = np.average(V_out[mid:end-1])

    I_in_ave = sum([(I_out[x] + I_out[x + 1]) / 2 * (time[x + 1] - time[x])
                 for x in range(start, end)]) / (time[end] - time[start])

    V_std = np.std(V_out[start:end-1]) 

    #print('P_out, P_in', P_out, P_in)
    #if P_in == 0:
    if P_in <0.001 and P_in>-0.001:
       P_in=0
       return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'power_in_is_zero'}
    if P_out <0.001 and P_out>-0.001:
        P_out = 0

    # process and visualize the results
    eff = P_out / (P_in+0.002)
    plt.figure(figsize=(25, 10))
    plt.subplot(2, 3, 1)
    topo_img = mpimg.imread(path[:-3] + "png")
    plt.imshow(topo_img, interpolation='kaiser')
    plt.title('topology')
    plt.subplot(2, 3, 2)
    plt.plot([time[x] for x in range(start, end)],
             [I_in[x] for x in range(start, end)])
    plt.title("I_in")
    plt.subplot(2, 3, 3)
    plt.plot([time[x] for x in range(start, end)],
             [I_out[x] for x in range(start, end)])
    plt.title("I_out")
    plt.subplot(2, 3, 4)
    plt.plot([time[x] for x in range(start, end)],
             [V_out[x] for x in range(start, end)])
    plt.title("V_out")
    plt.subplot(2, 3, 5)
    plt.plot([time[x] for x in range(1, end)],
             [V_out[x] for x in range(1, end)])
    plt.title("V_out all")
    plt.subplot(2, 3, 6)
    plt.plot([time[x] for x in range(1, end)],
             [I_out[x] for x in range(1, end)])
    plt.title("I_out all")

    plt.suptitle('efficiency({}): {:.3f} '.format(
              path[23:-4], eff),  
              horizontalalignment='right', fontsize=15)

    #stable_flag = V_std <= max(abs(V_out_ave*stable_ratio),V_in/50) and \
    #              (abs(V_out_ave_1-V_out_ave_2)<= max(abs(V_out_ave*stable_ratio),V_in/100))

    stable_flag = (abs(V_out_ave_1-V_out_ave_2)<= max(abs(V_out_ave*stable_ratio),V_in/200)) 

    #stable_flag = 1;

    result = {'result_valid': (0 <= eff <= 1) and stable_flag,              
              'efficiency': eff, 
              'error_msg': 'None',
              'output_voltage': V_out_ave}

    if stable_flag==0:
       
        case_id = simu_file.split('/')[1].split('.')[0]
        print('case_id', case_id)
        output_file_name = (
            'sim_analysis/errors/output_has_not_settled/'
            '{}.plot.png').format(case_id)
        result['error_msg'] = 'output_has_not_settled'

    elif eff < 0:
        case_id = simu_file.split('/')[1].split('.')[0]
        print('case_id', case_id)
        output_file_name = (
            'sim_analysis/errors/efficiency_is_less_than_zero/'
            '{}.plot.png').format(case_id)
        result['error_msg'] = 'efficiency_is_less_than_zero'
        
    elif eff > 1:
        case_id = simu_file.split('/')[1].split('.')[0]
        output_file_name = (
            'sim_analysis/errors/efficiency_is_greater_than_one/'
            '{}.plot.png').format(case_id)
        result['error_msg'] = 'efficiency_is_greater_than_one'
    else:
        case_id = simu_file.split('/')[1].split('.')[0]
        range_name = "{:.0f}_{:.0f}".format(
            math.floor(eff * 10) * 10, math.ceil(eff * 10) * 10)
        output_file_name = (
            'sim_analysis/valid/'
            '{}/{}.plot.png').format(range_name, case_id)
        
        if (V_out_ave < 0.6 * input_voltage or V_out_ave > 1.2 * input_voltage) and eff > 0.7:
            case_id = simu_file.split('/')[1].split('.')[0]
            print('case_id', case_id)
            output_file_name = (
                'sim_analysis/correct_candidates/'
                '{}.plot.png').format(case_id)
        
    print(output_file_name)
    tokens = output_file_name.split('/')
    prefix = "/".join(tokens[:-1])
    if not os.path.exists(prefix):
        os.system("mkdir -p " + prefix)
    plt.savefig(output_file_name)
    plt.close()
    return result


def calculate_efficiency_without_args(path, input_voltage, freq, sys_os="linux"):
    my_timeout = 5
    # info = 0  # this means no error occurred
    simu_file = path[:-3] + 'simu'
    file = open(simu_file, 'r')
    V_in = input_voltage
    V_out = []
    I_in = []
    I_out = []
    time = []
    stable_ratio = 0.01

    cycle = 1 / freq
    # count = 0

    read_V_out, read_I_out, read_I_in = False, False, False
    for line in file:
        # print(line)
        if "Transient solution failed" in line:
            return {'result_valid': False,
                    'efficiency': 0.0,
                    'error_msg': 'transient_simulation_failure'}
        if "Index   time            v(out)" in line and not read_V_out:
            read_V_out = True
            read_I_out = False
            read_I_in = False
            continue
        elif "Index   time            v(in_ext,in)" in line and not read_I_in:
            read_V_out = False
            read_I_out = False
            read_I_in = True
            continue
        elif "Index   time            v(out,out_ext)" in line and not read_I_out:
            read_V_out = False
            read_I_out = True
            read_I_in = False
            continue

        tokens = line.split()

        # print(tokens)
        if len(tokens) == 3 and tokens[0] != "Index":
            if read_V_out:
                try:
                    V_out.append(float(tokens[2]))
                except:
                    V_out.append(0)
                try:
                    time.append(float(tokens[1]))
                except:
                    time.append(0)
            elif read_I_in:
                try:
                    I_in.append(float(tokens[2]) / 0.001)
                except:
                    I_in.append(0)
            elif read_I_out:
                try:
                    I_out.append(float(tokens[2]) / 0.001)
                except:
                    I_out.append(0)

    # print(len(V_out),len(I_out),len(I_in),len(time))
    if len(V_out) == len(I_out) == len(I_in) == len(time):
        pass
    else:
        print("don't match")
        return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'output_is_not_aligned'}

    if not V_out or not I_in or not I_out:
        print(V_out)
        return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'missing_output_type'}

    # print(I_out, I_in)
    end = len(V_out) - 1
    start = len(V_out) - 1
    while start >= 0:
        if time[end] - time[start] >= 50 * cycle:
            break
        start -= 1

    if start == -1:
        print("duration less than one cycle")
        return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'less_than_one_cycle'}
    mid = int((start + end) / 2)

    # print(start, end,time[end] - time[start])
    P_in = sum([(I_in[x] + I_in[x + 1]) / 2 * (V_in + V_in) / 2 *
                (time[x + 1] - time[x])
                for x in range(start, end)]) / (time[end] - time[start])

    P_out = sum([(I_out[x] + I_out[x + 1]) / 2 * (V_out[x] + V_out[x + 1]) / 2 *
                 (time[x + 1] - time[x])
                 for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave = sum([(V_out[x] + V_out[x + 1]) / 2 * (time[x + 1] - time[x])
                     for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave_1 = np.average(V_out[start:mid])

    V_out_ave_2 = np.average(V_out[mid:end - 1])

    I_in_ave = sum([(I_out[x] + I_out[x + 1]) / 2 * (time[x + 1] - time[x])
                    for x in range(start, end)]) / (time[end] - time[start])

    V_std = np.std(V_out[start:end - 1])

    # print('P_out, P_in', P_out, P_in)
    # if P_in == 0:
    if P_in < 0.001 and P_in > -0.001:
        P_in = 0
        return {'result_valid': False,
                'efficiency': 0.0,
                'error_msg': 'power_in_is_zero'}
    if P_out < 0.001 and P_out > -0.001:
        P_out = 0

    # process and visualize the results
    eff = P_out / (P_in + 0.002)
    # plt.figure(figsize=(25, 10))
    # plt.subplot(2, 3, 1)
    # topo_img = mpimg.imread(path[:-3] + "png")
    # plt.imshow(topo_img, interpolation='kaiser')
    # plt.title('topology')
    # plt.subplot(2, 3, 2)
    # plt.plot([time[x] for x in range(start, end)],
    #          [I_in[x] for x in range(start, end)])
    # plt.title("I_in")
    # plt.subplot(2, 3, 3)
    # plt.plot([time[x] for x in range(start, end)],
    #          [I_out[x] for x in range(start, end)])
    # plt.title("I_out")
    # plt.subplot(2, 3, 4)
    # plt.plot([time[x] for x in range(start, end)],
    #          [V_out[x] for x in range(start, end)])
    # plt.title("V_out")
    # plt.subplot(2, 3, 5)
    # plt.plot([time[x] for x in range(1, end)],
    #          [V_out[x] for x in range(1, end)])
    # plt.title("V_out all")
    # plt.subplot(2, 3, 6)
    # plt.plot([time[x] for x in range(1, end)],
    #          [I_out[x] for x in range(1, end)])
    # plt.title("I_out all")
    #
    # plt.suptitle('efficiency({}): {:.3f} '.format(
    #     path[23:-4], eff),
    #     horizontalalignment='right', fontsize=15)

    # stable_flag = V_std <= max(abs(V_out_ave*stable_ratio),V_in/50) and \
    #              (abs(V_out_ave_1-V_out_ave_2)<= max(abs(V_out_ave*stable_ratio),V_in/100))

    stable_flag = (abs(V_out_ave_1 - V_out_ave_2) <= max(abs(V_out_ave * stable_ratio), V_in / 200))

    # stable_flag = 1;

    result = {'result_valid': (0 <= eff <= 1) and stable_flag,
              'efficiency': eff,
              'error_msg': 'None',
              'output_voltage': V_out_ave}

    if stable_flag == 0:

        case_id = simu_file.split('/')[1].split('.')[0]
        print('case_id', case_id)
        if sys_os == "windows":
            output_file_name = (
                'sim_analysis\\errors\\output_has_not_settled\\'
                '{}.plot.png').format(case_id)
        else:
            output_file_name = (
                'sim_analysis/errors/output_has_not_settled/'
                '{}.plot.png').format(case_id)
        result['error_msg'] = 'output_has_not_settled'

    elif eff < 0:
        case_id = simu_file.split('/')[1].split('.')[0]
        print('case_id', case_id)
        if sys_os == "windows":
            output_file_name = (
                'sim_analysis\\errors\\efficiency_is_less_than_zero\\'
                '{}.plot.png').format(case_id)
        else:
            output_file_name = (
                'sim_analysis/errors/efficiency_is_less_than_zero/'
                '{}.plot.png').format(case_id)
        result['error_msg'] = 'efficiency_is_less_than_zero'

    elif eff > 1:
        case_id = simu_file.split('/')[1].split('.')[0]
        if sys_os == "windows":
            output_file_name = (
                'sim_analysis\\errors\\efficiency_is_greater_than_one\\'
                '{}.plot.png').format(case_id)
        else:
            output_file_name = (
                'sim_analysis/errors/efficiency_is_greater_than_one/'
                '{}.plot.png').format(case_id)
        result['error_msg'] = 'efficiency_is_greater_than_one'
    else:
        case_id = simu_file.split('/')[1].split('.')[0]
        range_name = "{:.0f}_{:.0f}".format(
            math.floor(eff * 10) * 10, math.ceil(eff * 10) * 10)
        if sys_os == "windows":
            output_file_name = (
                'sim_analysis\\valid\\'
                '{}\\{}.plot.png').format(range_name, case_id)
        else:
            output_file_name = (
                'sim_analysis/valid/'
                '{}/{}.plot.png').format(range_name, case_id)

        if (V_out_ave < 0.6 * input_voltage or V_out_ave > 1.2 * input_voltage) and eff > 0.7:
            case_id = simu_file.split('/')[1].split('.')[0]
            print('case_id', case_id)
            if sys_os == "windows":
                output_file_name = (
                    'sim_analysis\\correct_candidates\\'
                    '{}.plot.png').format(case_id)
            else:
                output_file_name = (
                    'sim_analysis/correct_candidates/'
                    '{}.plot.png').format(case_id)

    print(output_file_name)
    if sys_os == "windows":
        tokens = output_file_name.split('\\')
        prefix = "\\".join(tokens[:-1])
    else:
        tokens = output_file_name.split('/')
        prefix = "/".join(tokens[:-1])

    if not os.path.exists(prefix):
        os.system("mkdir -p " + prefix)
    # plt.savefig(output_file_name)
    # plt.close()
    return result


def analysis_topologies(configs, num_topology, num_components, output_folder="component_data_random", output=False):
    efficiencies = []
    sys_os = configs['sys_os']
    start = time.time()
    directory_path = str(num_components) + output_folder
    res = []
    # category_count = defaultdict(int)
    category_count = {'efficiency_is_greater_than_one': 0, 'efficiency_is_less_than_zero': 0, \
                      'less_than_one_cycle': 0, 'missing_output_type': 0, 'output_is_not_aligned': 0, \
                      'power_in_is_zero': 0, 'output_has_not_settled': 0, 'transient_simulation_failure': 0}

    if sys_os == "windows":
        # windows command delete prior topo_analysis
        os.system("del /s sim_analysis\\*.png")
        os.system("del sim_analysis\\analysis_result.txt")

    analysis_path = "sim_analysis"

    if not os.path.exists(analysis_path):
        os.system("mkdir -p " + analysis_path)

    f = open("sim_analysis/" + str(os.getpid()) + "-analysis_result.txt", "w+")

    for i in range(num_topology):
        name = "PCC-" + str(os.getpid()) + "-" + format(i, '06d')
        print(name)
        file_path = directory_path + '/' + name + '.cki'

        effi = calculate_efficiency_without_args(file_path, int(configs["vin"]), int(configs["freq"]), sys_os)
        print('effi', effi, '\n')
        efficiencies.append(effi)

        f.write(name)
        f.write("\n")
        if effi['result_valid']:
            f.write("efficiency:" + str(int(effi['efficiency'] * 100)) + "%")
            f.write("\n")
            f.write("output voltage:" + str(int(effi['output_voltage'])))
            f.write("\n\n")
        else:
            f.write("error message:" + effi['error_msg'])
            f.write("\n\n")

        if effi['result_valid']:
            res.append(effi['efficiency'])
        else:

            print("sim_analysis/errors/{}/{}.png".format(
                effi['error_msg'], name
            ))
            category_count[effi['error_msg']] += 1

            if (effi['error_msg'] == 'efficiency_is_greater_than_one' or
                    effi['error_msg'] == 'efficiency_is_less_than_zero'):
                continue
            if sys_os == "windows":
                if not os.path.exists("sim_analysis\\errors\\{}\\".format(effi['error_msg'])):
                    os.system("mkdir -p " + "sim_analysis\\errors\\{}\\".format(effi['error_msg']))
                try:
                    copyfile("{}\\{}.png".format(directory_path, name),
                             "sim_analysis\\errors\\{}\\{}.png".format(
                                 effi['error_msg'], name))
                except:
                    print("No file")

            else:
                if not os.path.exists("sim_analysis/errors/{}/".format(effi['error_msg'])):
                    os.system("mkdir -p " + "sim_analysis/errors/{}/".format(effi['error_msg']))
                try:
                    copyfile("{}/{}.png".format(directory_path, name),
                             "sim_analysis/errors/{}/{}.png".format(
                                 effi['error_msg'], name))
                except:
                    print("No file")
    print(category_count)
    names = ['eff>1', 'eff<0', 'cycle<1', 'o/p type', 'not aligned', 'Pin=0', 'Ring', 'transit_error']
    values = list(category_count.values())

    # errors_distribution = [(names[i], values[i]) for i in range(0, len(names[i]))]
    print("Error distribution:")
    print(str(category_count))
    print("Total invalid data: " + str(sum(values)))
    print()
    f.write("Errors distribution:\n")
    f.write(str(category_count) + "\n")
    f.write("Total invalid data: " + str(sum(values)))
    f.write("\n\n")

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    valid_count = [[0, 0], [0, 0]]
    # valid_count = plt.hist(res, bins, histtype='bar', rwidth=0.95, ec='black')
    valid_distribution = [(int(valid_count[1][i] * 100), int(valid_count[0][i])) for i in range(0, len(valid_count[0]))]
    print("valid distribution:")
    print(valid_distribution)
    print("Total valid data: " + str(int(sum(valid_count[0]))))
    print()
    f.write("Valid distribution:\n")
    f.write(str(valid_distribution) + "\n")
    f.write("Total valid data: " + str(int(sum(valid_count[0]))))
    f.write("\n\n")

    return efficiencies






if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_components', type=int, default=4, help='specify the number of component')
    parser.add_argument('-n_topology', type=int, default=5,
                        help='specify the number of topology you want to generate')
    parser.add_argument('-output_folder', type=str, default="components_data_random",
                        help='specify the output folder path')
    parser.add_argument('-input_voltage', type=float, default=50,
                        help='specify input DC voltage')
    parser.add_argument('-freq', type=int, default=200000,
                        help='specify switching frequency')
    parser.add_argument('-os', type=str, default="linux",
                        help='operating system')
    
    args = parser.parse_args()
    print(args)

    num_topology = args.n_topology
    num_components = args.n_components
    start = time.time()
#    directory_path = str(args.n_components) + args.output_folder

    directory_path = args.output_folder

    res = []
    #category_count = defaultdict(int)
    category_count = {'efficiency_is_greater_than_one':0,'efficiency_is_less_than_zero':0, \
                      'less_than_one_cycle':0, 'missing_output_type': 0, 'output_is_not_aligned': 0, \
                      'power_in_is_zero': 0, 'output_has_not_settled':0, 'transient_simulation_failure': 0}
    
    if (args.os=="windows"):
        #windows command delete prior topo_analysis
        os.system("del /s sim_analysis\*.png")
        os.system("del sim_analysis\analysis_result.txt")


    analysis_path="sim_analysis"

    if  os.path.exists(analysis_path):
        os.system("rm -r " + analysis_path)
 
    if not os.path.exists(analysis_path):
        os.system("mkdir -p " + analysis_path)
 
    f= open("sim_analysis/analysis_result.txt","w+")

    data={};

    for i in range(num_topology):
        name = "PCC-" + format(i, '06d')
        print(name)
        file_path = directory_path + '/' + name + '.cki'
        effi = calculate_efficiency(file_path, args.input_voltage, args.freq)
        print('effi', effi,'\n')
        
        f.write(name)
        f.write("\n")
        if effi['result_valid']:
             data[name]=[effi['efficiency'],effi['output_voltage']]
             f.write("efficiency:"+str(int(effi['efficiency']*100))+"%")
             f.write("\n")
             f.write("output voltage:"+str(int(effi['output_voltage'])))
             f.write("\n\n")
        else:
            data[name]=[-1,-1]
            f.write("error message:"+effi['error_msg'])
            f.write("\n\n")
        
        if effi['result_valid']:
            res.append(effi['efficiency'])
        else:

            print("sim_analysis/errors/{}/{}.png".format(
                effi['error_msg'], name
            ))
            category_count[effi['error_msg']] += 1
            
            if (effi['error_msg'] == 'efficiency_is_greater_than_one' or
                    effi['error_msg'] == 'efficiency_is_less_than_zero'):
                continue
            if not os.path.exists("sim_analysis/errors/{}/".format(effi['error_msg'])):
                os.system("mkdir -p " + "sim_analysis/errors/{}/".format(effi['error_msg']))
            copyfile("{}/{}.png".format(directory_path, name),
                     "sim_analysis/errors/{}/{}.png".format(
                         effi['error_msg'], name))


    print(category_count)                 
    names = ['eff>1','eff<0','cycle<1','o/p type','not aligned','Pin=0','Ring','transit_error']
    values = list(category_count.values())


    #errors_distribution = [(names[i], values[i]) for i in range(0, len(names[i]))] 
    print("Error distribution:")
    print(str(category_count))
    print("Total invalid data: "+str(sum(values)))
    print()
    f.write("Errors distribution:\n")
    f.write(str(category_count)+"\n")
    f.write("Total invalid data: "+str(sum(values)))
    f.write("\n\n")
 

    
    plt.figure(figsize=(10, 25))
    plt.subplot(2, 1, 1)
    plt.title('efficiency')
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    valid_count=plt.hist(res, bins, histtype='bar', rwidth=0.95, ec='black')
    valid_distribution = [(int(valid_count[1][i]*100), int(valid_count[0][i])) for i in range(0, len(valid_count[0]))] 
    print("valid distribution:")
    print(valid_distribution)
    print("Total valid data: "+str(int(sum(valid_count[0]))))
    print()
    f.write("Valid distribution:\n")
    f.write(str(valid_distribution)+"\n")
    f.write("Total valid data: "+str(int(sum(valid_count[0]))))
    f.write("\n\n")
    f.close()
    
    plt.subplot(2, 1, 2)
    plt.title('distribution')
    plt.bar(range(len(names)), values, tick_label=names) 
    plt.suptitle('Efficiency & error distribution')

    fig = plt.gcf()
    fig.set_size_inches(12, 8)

    plt.savefig('sim_analysis/summary.png')

    plt.show()

    with open("sim_analysis/result.json",'w') as outfile:
        json.dump(data,outfile)

    outfile.close()


    

    

