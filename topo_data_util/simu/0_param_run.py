import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from collections import defaultdict
from shutil import copyfile
import os
import subprocess
from subprocess import call

n_components="4"
n_topology="4000"
output_folder="component_data_random"
freq = "200000"
vin = "50"
D = "0.48"


my_system="linux" #linux or windows

ngspice_para="\"freq=%s vin=%s rload=50 iout=0.2 rcap=1Meg rind=0.1 rvinout=0.1 D=%s\"" % (freq, vin, D)
 

if my_system == "windows":
      
      ngspice_para="\"freq=%s vin=%s rload=50 iout=0.2 rcap=1Meg rind=0.1 rvinout=0.1 D=%s\"" % (freq, vin, D)
 
      cap_paras="\"1, 10\""# or "10,20"
      ind_paras="\"1, 10\"" # or ,"30,50"
    
      cmd_topology = "python3" + " 1_build_topology.py" + " -n_components=" \
      + n_components + " -n_topology=" + n_topology + " -output_folder=" \
      + output_folder + " -ngspice_para=" + ngspice_para + " -cap_paras="  +cap_paras \
      + " -ind_paras=" + ind_paras  

      print("Generate topology: " + cmd_topology + "\n")

      os.system(cmd_topology)

      cmd_simulation = "python3"+" 2_simulation.py" \
      + " -n_components=" + n_components \
      + " -n_topology=" + n_topology \
      + " -output_folder=" + output_folder\
      + " -os=" + my_system

      print("Simulate circuits: " + cmd_simulation + "\n")

      os.system(cmd_simulation)

      cmd_analysis = "python3"+" 3_simulation_analysis.py" \
      + " -n_components=" + n_components \
      + " -n_topology=" + n_topology \
      + " -output_folder=" + output_folder \
      + " -input_voltage=" + vin \
      + " -freq=" + freq

      print("Analyze simulation: " + cmd_analysis + "\n")

      os.system(cmd_analysis)


else:

    cap_paras="1,5,20,50"# or "10,20"
    ind_paras="1,10,50,150" # or ,"30,50"
    
    ngspice_para="freq=%s vin=%s rload=50 iout=0.2 rcap=1Meg rind=0.1 rvinout=0.1 D=%s" % (freq, vin, D)

    start1=time.time()

    call(["python3","1_build_topology.py","-n_components=" + n_components, \
    "-n_topology=" + n_topology, \
    "-output_folder=" + output_folder, \
    "-ngspice_para=" + ngspice_para, \
    "-cap_paras=" + cap_paras, \
    "-ind_paras=" + ind_paras, \
    "-os=" + my_system \
    ])

    start2=time.time()


    call(["python3","2_simulation.py", \
    "-n_components=" + n_components, \
    "-n_topology=" + n_topology, \
    "-output_folder=" + output_folder, \
    "-os=" + my_system\
    ])

    start3=time.time()

    call(["python3","3_simulation_analysis.py", \
    "-n_components=" + n_components, \
    "-n_topology=" + n_topology, \
    "-output_folder=" + output_folder, \
    "-input_voltage=" + vin, \
    "-freq=" + freq,\
    "-os=" + my_system
    ])
    
    end=time.time()


    cmd_topology = "python3" + " 1_build_topology.py" + " -n_components=" \
    + n_components + " -n_topology=" + n_topology + " -output_folder=" \
    + output_folder + " -ngspice_para=" + "\"" + ngspice_para + "\""+ " -cap_paras="  +cap_paras \
    + " -ind_paras=" + ind_paras + " -os=" + my_system  

    print("Generate topology: \n" + cmd_topology + "\n")

    cmd_simulation = "python3"+" 2_simulation.py" \
    + " -n_components=" + n_components \
    + " -n_topology=" + n_topology \
    + " -output_folder=" + output_folder\
    + " -os=" + my_system

    print("Simulate circuits: \n" + cmd_simulation + "\n")

    cmd_analysis = "python3"+" 3_simulation_analysis.py" \
    + " -n_components=" + n_components \
    + " -n_topology=" + n_topology \
    + " -output_folder=" + output_folder \
    + " -input_voltage=" + vin \
    + " -freq=" + freq \
    + " -os=" + my_system

    print("Analyze simulation: \n" + cmd_analysis + "\n")

    print("Topology generation time: "+str(start2-start1)+"\n")
    print("Simulation  time: "+str(start3-start2)+"\n")
    print("Analysis time: "+str(end-start3)+"\n")



    f=open("command.txt","w+")
    f.write(cmd_topology)
    f.write("\n\n")
    f.write(cmd_simulation)
    f.write("\n\n")
    f.write(cmd_analysis)
    f.write("\n\n")
    f.close()




# python 1_build_topology.py \
# -n_components=4 \
# -n_topology=5 \
# -output_folder=component_data_random \
# -ngspice_para="freq=200k vin=5 rload=10 iout=0.2 rcap=1Meg rind=0.1 rvinout=0.1 D=0.5" \
# -cap_paras=10 \
# -ind_paras=100

# python3 2_simulation.py \
# -n_components=4 \
# -n_topology=5 \
# -output_folder=component_data_random \
# -os=mac

# python3 3_simulation_analysis.py \
# -n_components=4 \
# -n_topology=5 \
# -output_folder=component_data_random


