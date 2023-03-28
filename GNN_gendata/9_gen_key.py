from gen_topo import *

if __name__ == '__main__':

       cki=json.load(open("database/circuit.json"))
       parameters = json.load(open("param.json"))

       data=json.load(open("database/data.json"))
       result=json.load(open("../dataset_5.json"))

       hashtable={}
       hashtable_flag={}

       nn=0

       for fn in result:
           print(nn,' ',fn)
           nn=nn+1
           name=fn[0:11]
           key=data[name]['key']
           duty_cycle=result[fn]['duty_cycle']
           eff=result[fn]['eff']
           vout=result[fn]['vout']
           vout_flag=int(vout>30 and vout<70)
           key_total=key+'$'+str(duty_cycle)
           result_list=[eff,vout]
           result_list_flag=[eff,vout_flag]
           hashtable[key_total]=result_list
           hashtable_flag[key_total]=result_list_flag

       with open('../hashtable.json','w') as f:
          json.dump(hashtable,f)
       f.close()

       with open('../hashtable_flag.json','w') as f:
          json.dump(hashtable_flag,f)
       f.close()


