from gen_topo import *

if __name__ == '__main__':

    analytic = json.load(open("database/analytic.json"))
    cki_folder = 'database/cki/'
    data = {}
    data_csv = []
    for fn in analytic:
        count = 0
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = fn + '-' + str(count) + '.simu'
            path = cki_folder + file_name
            vin = param_value[device_name['Vin']]
            freq = param_value[device_name['Frequency']] * 1000000
            rin = param_value[device_name['Rin']]
            rout = param_value[device_name['Rout']]
            print(file_name)
            result = calculate_efficiency(path, vin, freq, rin, rout)

            if not result['result_valid']:
                print(result['error_msg'])

            VO = int(result["Vout"])
            E = int(result["efficiency"] * 100)
            flag_candidate = (VO < vin * 0.7 or VO > vin * 1.2) and E > 70

            data[fn] = {}
            data[fn][param] = [device_name, netlist, VO, E, flag_candidate]
            tmp = [fn]
            for item in param_value:
                tmp.append(item)
            tmp.append(VO)
            tmp.append(E)
            data_csv.append(tmp)
            count = count + 1
    # print(result)
    with open("./database/sim_result_csv.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_csv)
    f.close()

    with open('./database/sim_result.json', 'w') as f:
        json.dump(data, f)
    f.close()

















