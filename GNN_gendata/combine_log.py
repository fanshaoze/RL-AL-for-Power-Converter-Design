from lcapy import Circuit
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import json
import time as time
import os
import csv

if __name__ == '__main__':
                
    data_csv=[]
 
    json_total={}

    for i in range(16):
           
              file_path = "./t"+format(i,'02d')+"/out.log"
              print(file_path)
              f=open(file_path,'r')
              lines = f.readlines()

              for line in lines:
                  if line[2:6]=='Topo':
                     tmp=line[1:-2].split(", ")
                     tmp_csv=[]
                     tmp_csv.append(tmp[0])
                     tmp_csv.append(tmp[1])
                     tmp_csv.append(tmp[-3])
                     tmp_csv.append(tmp[-2])
                     tmp_csv.append(tmp[-1])
#                     data_csv.append(tmp_csv)

                     name=tmp[0][1:-1]
                    # param="("+tmp[1]+", 1, 100, 0.1, 100, 1e-05, 0.0001, 1, 100000, 0.0001, 1e-05, 1, 100000)"

                     try:
                        param1=tmp[1]
                        param2=tmp[5]
                        param=str(param1)+'-'+str(param2)
                     except:
                         continue

                     if tmp[-3]=='False' or tmp[-2]=='False' or tmp[1]=='False':
                         continue
                     
                     print(name)
                     print(param)

                     data_csv.append([name,param1,param2,int(tmp[-3]),int(tmp[-2])/100])

                     try:
                       json_total[name][param]={}
                     except:
                       json_total[name]={}
                       json_total[name][param]={}
                     try:
                        json_total[name][param]["vout"]=int(tmp[-3])
                        json_total[name][param]["eff"]=int(tmp[-2])/100
                     except:
                        json_total[name][param]["vout"]=0
                        json_total[name][param]["eff"]=0


    with open("./sim.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_csv)
    f.close()

    with open("./sim.json","w") as f1:
        json.dump(json_total,f1)
    f1.close()


