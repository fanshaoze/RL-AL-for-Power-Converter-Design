from lcapy import Circuit
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import json
from SimulatorAnalysis.gen_topo import *
import csv


def assign_DC_C_and_L_in_param(param, fix_paras):
    assert fix_paras['C'] != []
    assert fix_paras['L'] != []
    assert fix_paras['Duty_Cycle'] != []
    param['Duty_Cycle'] = fix_paras['Duty_Cycle']
    param['C'] = fix_paras['C']
    param['L'] = fix_paras['L']
    return param


def get_analytics_information(target_folder, name, fix_paras, expression=None):
    if expression:
        exp = expression
    else:
        exp = json.load(open(target_folder + '/' + name + '-expr_val.json'))
    # parameters = json.load(open("param.json"))
    parameters = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    assert fix_paras is not None
    parameters = assign_DC_C_and_L_in_param(parameters, fix_paras)
    target_folder = './'

    valid_topo = []

    result = {}
    result_csv = []

    count = 0
    count_max = 300

    for fn in exp:

        if exp[fn] == {}:

            result[fn] = {}

            for item in parameters['Duty_Cycle']:
                tmp = [fn, str(item), False, False, False, False]
                result_csv.append(tmp)
                result[fn][str(item)] = [False, False, False, False, False]

            continue

        net_list = exp[fn]['net_list']

        count = count + 1

        if count > count_max:
            break

        name = fn

        a_x = exp[fn]['A state']['x']
        a_y = exp[fn]['A state']['y']
        a_A = exp[fn]['A state']['a']
        a_B = exp[fn]['A state']['b']
        a_C = exp[fn]['A state']['c']
        a_D = exp[fn]['A state']['d']

        b_x = exp[fn]['B state']['x']
        b_y = exp[fn]['B state']['y']
        b_A = exp[fn]['B state']['a']
        b_B = exp[fn]['B state']['b']
        b_C = exp[fn]['B state']['c']
        b_D = exp[fn]['B state']['d']

        device_list = exp[fn]['device_list']

        param2sweep, paramname = gen_param(device_list, parameters)

        paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

        name_list = {}
        for index, name in enumerate(paramname):
            name_list[name] = index

        result[fn] = {}

        for vect in paramall:
            # print(fn, vect)
            duty_cycle = vect[name_list['Duty_Cycle']]
            vin = vect[name_list['Vin']]
            rin = vect[name_list['Rin']]
            rout = vect[name_list['Rout']]
            freq = vect[name_list['Frequency']]
            cout = vect[name_list['Cout']]

            a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = a_x, a_y, a_A, a_B, a_C, a_D
            b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = b_x, b_y, b_A, b_B, b_C, b_D

            nodelist = a_y[2:-2].split('], [')
            statelist = a_x[2:-2].split('], [')

            k = 0
            for node in nodelist:

                if str(node) == 'v_IN(t)':
                    Ind_Vin = k
                    flag_IN = 1
                if str(node) == 'v_IN_exact(t)':
                    Ind_Vinext = k
                    flag_IN_ext = 1
                if str(node) == 'v_OUT(t)':
                    Ind_Vout = k
                    flag_OUT = 1
                k = k + 1

            if flag_IN * flag_IN_ext * flag_OUT == 0:
                print('can not find voltage node')
                continue

            for index, value in enumerate(vect):
                a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = exp_subs(a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp, paramname[index],
                                                              str(value))
                b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = exp_subs(b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp, paramname[index],
                                                              str(value))

            A = duty_cycle * np.array(eval(a_Ap)) + (1 - duty_cycle) * np.array(eval(b_Ap))
            B = duty_cycle * np.array(eval(a_Bp)) + (1 - duty_cycle) * np.array(eval(b_Bp))
            C = duty_cycle * np.array(eval(a_Cp)) + (1 - duty_cycle) * np.array(eval(b_Cp))
            D = duty_cycle * np.array(eval(a_Dp)) + (1 - duty_cycle) * np.array(eval(b_Dp))
            try:
                A_inv = np.linalg.inv(A)

            except:
                tmp = []
                tmp.append(fn)
                tmp.append(str(duty_cycle))
                tmp.append(str(vect))
                tmp.append(False)
                tmp.append(False)
                tmp.append(False)
                result_csv.append(tmp)
                result[fn][str(vect)] = [name_list, net_list, False, False, False]

                continue

            x_static = -np.matmul(A_inv, B) * vin
            y_static = (-np.matmul(np.matmul(C, A_inv), B) + D) * vin

            Vout = y_static[Ind_Vout]
            Iin = abs((y_static[Ind_Vin] - y_static[Ind_Vinext]) / rin)
            Pin = Iin * vin
            Pout = Vout * Vout / rout
            eff = Pout / (Pin + 0.01)

            VO = int(Vout[0])
            E = float(int(eff[0] * 100)) / 100
            flag_candidate = (VO > -500) and (VO < 500) and (VO < vin * 0.6 or VO > vin * 1.2) and 60 < E < 100

            tmp = [fn, duty_cycle, str(vect), VO, E, flag_candidate]

            result_csv.append(tmp)
            result[fn][str(vect)] = [name_list, net_list, VO, E, flag_candidate, 1]

    # print(result)
    # path = target_folder + '/' + name + '-analytic_csv.csv'
    # append_csv(path, result_csv)

    # with open(target_folder + '/' + name + '-analytic.json', 'w') as f:
    #     json.dump(result, f)
    # f.close()
    return result


def get_analytics_result(target_folder, name, fix_paras, expression=None):
    if expression:
        exp = expression
    else:
        exp = json.load(open(target_folder + '/' + name + '-expr_val.json'))
    # parameters = json.load(open("param.json"))
    parameters = json.load(open("./UCT_5_UCB_unblc_restruct_DP_v1/SimulatorAnalysis/param.json"))
    assert fix_paras is not None
    parameters = assign_DC_C_and_L_in_param(parameters, fix_paras)
    print('analytics parameter', parameters)

    target_folder = './'

    valid_topo = []

    result = {}
    result_csv = []

    count = 0
    count_max = 300

    for fn in exp:

        if exp[fn] == {}:

            result[fn] = {}

            for item in parameters['Duty_Cycle']:
                tmp = [fn, str(item), False, False, False, False]
                result_csv.append(tmp)
                result[fn][str(item)] = [False, False, False, False, False]

            continue

        net_list = exp[fn]['net_list']

        count = count + 1

        if count > count_max:
            break

        name = fn

        a_x = exp[fn]['A state']['x']
        a_y = exp[fn]['A state']['y']
        a_A = exp[fn]['A state']['a']
        a_B = exp[fn]['A state']['b']
        a_C = exp[fn]['A state']['c']
        a_D = exp[fn]['A state']['d']

        b_x = exp[fn]['B state']['x']
        b_y = exp[fn]['B state']['y']
        b_A = exp[fn]['B state']['a']
        b_B = exp[fn]['B state']['b']
        b_C = exp[fn]['B state']['c']
        b_D = exp[fn]['B state']['d']

        device_list = exp[fn]['device_list']

        param2sweep, paramname = gen_param(device_list, parameters)

        paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

        name_list = {}
        for index, name in enumerate(paramname):
            name_list[name] = index

        result[fn] = {}

        for vect in paramall:
            # print(fn, vect)
            duty_cycle = vect[name_list['Duty_Cycle']]
            vin = vect[name_list['Vin']]
            rin = vect[name_list['Rin']]
            rout = vect[name_list['Rout']]
            freq = vect[name_list['Frequency']]
            cout = vect[name_list['Cout']]

            a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = a_x, a_y, a_A, a_B, a_C, a_D
            b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = b_x, b_y, b_A, b_B, b_C, b_D

            nodelist = a_y[2:-2].split('], [')
            statelist = a_x[2:-2].split('], [')

            k = 0
            for node in nodelist:

                if str(node) == 'v_IN(t)':
                    Ind_Vin = k
                    flag_IN = 1
                if str(node) == 'v_IN_exact(t)':
                    Ind_Vinext = k
                    flag_IN_ext = 1
                if str(node) == 'v_OUT(t)':
                    Ind_Vout = k
                    flag_OUT = 1
                k = k + 1

            if flag_IN * flag_IN_ext * flag_OUT == 0:
                print('can not find voltage node')
                continue

            for index, value in enumerate(vect):
                a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = exp_subs(a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp, paramname[index],
                                                              str(value))
                b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = exp_subs(b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp, paramname[index],
                                                              str(value))

            A = duty_cycle * np.array(eval(a_Ap)) + (1 - duty_cycle) * np.array(eval(b_Ap))
            B = duty_cycle * np.array(eval(a_Bp)) + (1 - duty_cycle) * np.array(eval(b_Bp))
            C = duty_cycle * np.array(eval(a_Cp)) + (1 - duty_cycle) * np.array(eval(b_Cp))
            D = duty_cycle * np.array(eval(a_Dp)) + (1 - duty_cycle) * np.array(eval(b_Dp))
            try:
                A_inv = np.linalg.inv(A)

            except:
                tmp = []
                tmp.append(fn)
                tmp.append(str(duty_cycle))
                tmp.append(str(vect))
                tmp.append(False)
                tmp.append(False)
                tmp.append(False)
                result_csv.append(tmp)
                result[fn][str(vect)] = [name_list, net_list, False, False, False]

                continue

            x_static = -np.matmul(A_inv, B) * vin
            y_static = (-np.matmul(np.matmul(C, A_inv), B) + D) * vin

            Vout = y_static[Ind_Vout]
            Iin = abs((y_static[Ind_Vin] - y_static[Ind_Vinext]) / rin)
            Pin = Iin * vin
            Pout = Vout * Vout / rout
            eff = Pout / (Pin + 0.01)

            VO = int(Vout[0])
            E = float(int(eff[0] * 100)) / 100
            flag_candidate = (VO > -500) and (VO < 500) and (VO < vin * 0.6 or VO > vin * 1.2) and 60 < E < 100

            tmp = [fn, duty_cycle, str(vect), VO, E, flag_candidate]

            result_csv.append(tmp)
            result[fn][str(vect)] = [name_list, net_list, VO, E, flag_candidate, 1]

    path = target_folder + '/' + name + '-analytic_csv.csv'
    append_csv(path, result_csv)

    with open(target_folder + '/' + name + '-analytic.json', 'w') as f:
        json.dump(result, f)
    f.close()
    return result_csv


def append_csv(path, datas):
    with open(path, "a+", newline='') as file:
        csv_file = csv.writer(file)
        csv_file.writerows(datas)
    file.close()


if __name__ == '__main__':

    exp = json.load(open("database/expression.json"))
    parameters = json.load(open("param.json"))

    target_folder = './'

    valid_topo = []

    result = {}
    result_csv = []

    count = 0
    count_max = 100000

    for fn in exp:

        net_list = exp[fn]['net_list']

        count = count + 1

        if count > count_max:
            break

        name = fn

        a_x = exp[fn]['A state']['x']
        a_y = exp[fn]['A state']['y']
        a_A = exp[fn]['A state']['a']
        a_B = exp[fn]['A state']['b']
        a_C = exp[fn]['A state']['c']
        a_D = exp[fn]['A state']['d']

        b_x = exp[fn]['B state']['x']
        b_y = exp[fn]['B state']['y']
        b_A = exp[fn]['B state']['a']
        b_B = exp[fn]['B state']['b']
        b_C = exp[fn]['B state']['c']
        b_D = exp[fn]['B state']['d']

        device_list = exp[fn]['device_list']

        param2sweep, paramname = gen_param(device_list, parameters)

        paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

        name_list = {}
        for index, name in enumerate(paramname):
            name_list[name] = index

        result[fn] = {}

        for vect in paramall:
            # print(fn, vect)
            duty_cycle = vect[name_list['Duty_Cycle']]
            vin = vect[name_list['Vin']]
            rin = vect[name_list['Rin']]
            rout = vect[name_list['Rout']]
            freq = vect[name_list['Frequency']]
            cout = vect[name_list['Cout']]

            a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = a_x, a_y, a_A, a_B, a_C, a_D
            b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = b_x, b_y, b_A, b_B, b_C, b_D

            nodelist = a_y[2:-2].split('], [')
            statelist = a_x[2:-2].split('], [')

            k = 0
            for node in nodelist:

                if str(node) == 'v_IN(t)':
                    Ind_Vin = k
                    flag_IN = 1
                if str(node) == 'v_IN_exact(t)':
                    Ind_Vinext = k
                    flag_IN_ext = 1
                if str(node) == 'v_OUT(t)':
                    Ind_Vout = k
                    flag_OUT = 1
                k = k + 1

            if flag_IN * flag_IN_ext * flag_OUT == 0:
                print('can not find voltage node')
                continue

            for index, value in enumerate(vect):
                a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = exp_subs(a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp, paramname[index],
                                                              str(value))
                b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = exp_subs(b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp, paramname[index],
                                                              str(value))

            A = duty_cycle * np.array(eval(a_Ap)) + (1 - duty_cycle) * np.array(eval(b_Ap))
            B = duty_cycle * np.array(eval(a_Bp)) + (1 - duty_cycle) * np.array(eval(b_Bp))
            C = duty_cycle * np.array(eval(a_Cp)) + (1 - duty_cycle) * np.array(eval(b_Cp))
            D = duty_cycle * np.array(eval(a_Dp)) + (1 - duty_cycle) * np.array(eval(b_Dp))
            try:
                A_inv = np.linalg.inv(A)

            except:
                tmp = []
                tmp.append(fn)
                tmp.append(str(vect))
                tmp.append(False)
                tmp.append(False)
                tmp.append(False)
                result_csv.append(tmp)
                result[fn][str(vect)] = [name_list, net_list, False, False, False, 0]

                continue

            x_static = -np.matmul(A_inv, B) * vin
            y_static = (-np.matmul(np.matmul(C, A_inv), B) + D) * vin

            Vout = y_static[Ind_Vout]
            Iin = abs((y_static[Ind_Vin] - y_static[Ind_Vinext]) / rin)
            Pin = Iin * vin
            Pout = Vout * Vout / rout
            eff = Pout / (Pin + 0.01)

            VO = int(Vout[0])
            E = int(eff[0] * 100)
            flag_candidate = (VO > -500) and (VO < 500) and (VO < vin * 0.7 or VO > vin * 1.2) and 70 < E < 100

            tmp = [fn, str(vect), VO, E, flag_candidate, 1]

            result_csv.append(tmp)
            result[fn][str(vect)] = [name_list, net_list, VO, E, flag_candidate, 1]

    # print(result)
    with open("./database/analytic.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(result_csv)
    f.close()

    with open('./database/analytic.json', 'w') as f:
        json.dump(result, f)
    f.close()
