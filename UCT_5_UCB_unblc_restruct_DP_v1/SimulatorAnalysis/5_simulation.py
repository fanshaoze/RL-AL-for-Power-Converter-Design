from gen_topo import *

if __name__ == '__main__':

    analytic = json.load(open("database/analytic.json"))
    cki_folder = 'database/cki/'
    for fn in analytic:
        if int(fn[6:9]) > 512:
            print(int(fn[6:9]))
            count = 0
            for param in analytic[fn]:
                param_value = [float(value) for value in param[1:-1].split(', ')]
                device_name = analytic[fn][param][0]
                netlist = analytic[fn][param][1]
                file_name = fn + '-' + str(count) + '.cki'
                convert_cki(file_name, param_value, device_name, netlist)
                print(file_name)
                simulate(cki_folder + file_name)
                count = count + 1
