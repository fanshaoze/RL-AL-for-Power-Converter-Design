import os
import time
import subprocess
import signal
import argparse
debug = False
def simulate(path):
    my_timeout = 10
    info = 0  # this means no error occurred

############ run ngspice in windows ###################

    if args.os == "windows":
        cmd = "D:/NGspice/Spice64/bin/ngspice"
        simu_file = path[:-3] + 'simu'
        p=subprocess.run([cmd,"-b", path , "-o" , simu_file])

        
############ run ngspice in windows ###################

############ run ngspice in Mac os ####################
    elif args.os == "mac":

        simu_file = path[:-3] + 'simu'
        p = subprocess.Popen("exec " + 'ngspice -b ' + path + '>' + simu_file, stdout=subprocess.PIPE, shell=True)
        try:
            p.wait(my_timeout)
        except subprocess.TimeoutExpired:
            print("kill\n")
            p.kill()

############ run ngspice in linux os ####################
    elif args.os == "linux":

        simu_file = path[:-3] + 'simu'
        p = subprocess.Popen("exec " + 'ngspice -b ' + path + '>' + simu_file, stdout=subprocess.PIPE, shell=True)
        try:
            p.wait(my_timeout)
        except subprocess.TimeoutExpired:
            print("kill\n")
            p.kill()


def simulate_without_agrs(path, sys_os="linux"):
    my_timeout = 5
    info = 0  # this means no error occurred

############ run ngspice in windows ###################

    if sys_os == "windows":
        # cmd = "D:/NGspice/Spice64/bin/ngspice"
        cmd = "C:/1D/Spice64/bin/ngspice"
        simu_file = path[:-3] + 'simu'
        p=subprocess.run([cmd,"-b", path , "-o" , simu_file])


############ run ngspice in windows ###################

############ run ngspice in Mac os ####################
    elif sys_os == "mac":

        simu_file = path[:-3] + 'simu'
        p = subprocess.Popen("exec " + 'ngspice -b ' + path + '>' + simu_file, stdout=subprocess.PIPE, shell=True)
        try:
            p.wait(my_timeout)
        except subprocess.TimeoutExpired:
            print("kill\n")
            p.kill()

############ run ngspice in linux os ####################
    elif sys_os == "linux":

        simu_file = path[:-3] + 'simu'
        p = subprocess.Popen("exec " + 'ngspice -b ' + path + '>' + simu_file, stdout=subprocess.PIPE, shell=True)
        try:
            p.wait(my_timeout)
        except subprocess.TimeoutExpired:
            print("kill\n")
            p.kill()

def simulate_topologies(num_topology, num_components, sys_os="linux", output_folder="component_data_random"):
    directory_path = str(num_components) + output_folder
    if sys_os == "windows":
    ############# windows command to delete pervious data #################
        cmd = "cd "
        # folder_path = '\"D:\\Dropbox\\1. Research\\1_2020_RLPM\\4_Parasitic_Model\\'+directory_path+'\"'
        folder_path = '\"C:\\1D\\pypro\\UCF\\topo_new\\'+directory_path+'\"'
        print(cmd + folder_path)
        print("del *")
    ############# windows command to delete pervious data #################

    for i in range(num_topology):

        # name = "PCC-" + format(i+1483, '06d')
        name = "PCC-" + str(os.getpid()) + "-" + format(i, '06d')
        print(name)
        file_path = directory_path + '/' + name + '.cki'

        simulate_without_agrs(file_path, sys_os)


############ run ngspice in Mac os ####################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n_components', type=int, default=4, help='specify the number of component')
    parser.add_argument('-n_topology', type=int, default=5,
                        help='specify the number of topology you want to generate')
    parser.add_argument('-output_folder', type=str, default="components_data_random",
                        help='specify the output folder path')
    parser.add_argument('-os', type=str, default="mac",
                        help='mac, windows')
    args = parser.parse_args()
    print(args)


    num_topology = args.n_topology
    num_components = args.n_components
    start = time.time()
#    directory_path = str(args.n_components) + args.output_folder
    directory_path = args.output_folder

    if (args.os=="windows"):
    ############# windows command to delete pervious data #################
        cmd = "cd "
        folderpath = '\"D:\\Dropbox\\1. Research\\1_2020_RLPM\\4_Parasitic_Model\\'+directory_path+'\"'
        print(cmd + folderpath)
        print("del *")
    ############# windows command to delete pervious data #################

    for i in range(num_topology):

        name = "PCC-" + format(i, '06d')
        print(name)
        file_path = directory_path + '/' + name + '.cki'

        simulate(file_path)

        # print(info)

    elapsed_time_secs = time.time() - start

    msg = 'AVG simulation took in secs: ', (elapsed_time_secs*0.1/num_topology)

    print(msg)

