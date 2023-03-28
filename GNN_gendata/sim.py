from gen_topo import *
import os


if __name__ == '__main__':

   cki=json.load(open("../../database/circuit.json"))
   parameters = json.load(open("../../param.json"))

   path = os.path.abspath(os.getcwd())

   folder_ind=int(path[-2:])
#   folder_ind=0

   print(folder_ind)

   data_csv=[]

   cnt=-1


   cki_folder='./cki/'
   for fn in cki:

        cnt+=1
        if cnt>10000:
            continue

        if cnt%16!=folder_ind:
            continue

        device_list=["Duty_Cycle","Frequency"]+cki[fn]["device_list"]

       
        param2sweep,paramname=gen_param(device_list,parameters)

        paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

        name_list={}
        for index,name in enumerate(paramname):
            name_list[name]=index


        count=0

        device_name={}
        
        for i,item in enumerate(name_list):
            device_name[item]=i


        for vect in paramall:
               print(fn+'-'+str(count))
               netlist=cki[fn]["net_list"]
               file_name='simulation.cki'
               convert_cki(file_name,vect,device_name,netlist)
               simulate(cki_folder+file_name)

               param_value=vect
               path=cki_folder+file_name[:-3]+'simu'
               duty_cycle=param_value[device_name['Duty_Cycle']] 
               vin=param_value[device_name['Vin']]
               freq=param_value[device_name['Frequency']]*1000000
               rin=param_value[device_name['Rin']]
               rout=param_value[device_name['Rout']]
               result=calculate_efficiency(path,vin,freq,rin,rout)

   
               if result['result_valid']==False:
                   tmp=[]
                   tmp.append(fn)
                   tmp.append(duty_cycle)
                   tmp.append(str(vect))
                   tmp.append(0)
                   tmp.append(0)
                   tmp.append(False)
                   print(tmp)
                   print('\n')
                   data_csv.append(tmp)
#                   data[fn]={}
#                   data[fn][str(vect)]=[False,False,False,False,False] 
                   count=count+1
                   continue
   
   
               VO=int(result["Vout"])
               E=int(result["efficiency"]*100)
               flag_candidate=(VO<vin*0.7 or VO>vin*0.3) and E>70
   
#               data[fn]={}
#               data[fn][str(vect)]=[device_name,netlist,VO,E,flag_candidate] 
               tmp=[]
               tmp.append(fn)
               tmp.append(duty_cycle)
               tmp.append(str(vect))
               tmp.append(VO)
               tmp.append(E)
               tmp.append(flag_candidate)
               print(tmp)
               data_csv.append(tmp)


               print('\n')
