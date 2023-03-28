from GNN_gendata.tr_random_topo import *
from GNN_gendata.tr_data import *
from GNN_gendata.tr_dataset import *
from GNN_gendata.tr_circuit import *


def get_one_topo_train_data(current, efficiency, vout, anal_efficiency=0, anal_vout=0):
    init_data = gen_init_data(current)
    if init_data is None:
        return None, None
    key, key_data = get_key_data(init_data)
    circuit_data = get_circuit_data(key_data)
    one_dataset = get_dataset(fix_paras={'Duty_Cycle': [current.parameters[0]],
                                         'C': [current.parameters[1]],
                                         'L': [current.parameters[2]]},
                              init_data=init_data, key_data=key_data, circuit_data=circuit_data,
                              eff=efficiency, vout=vout, eff_analytic=anal_efficiency, vout_analytic=anal_vout)
    return key, one_dataset


def update_dataset(current_set, training_date_file='./database/dataset_5.json'):
    dataset = json.load(open(training_date_file))
    for key_with_para, current_info in current_set.items():
        # one_dataset: current, efficiency, vout, anal_efficiency=0, anal_vout
        if key_with_para not in dataset:
            key, one_dataset = get_one_topo_train_data(current_info[0], current_info[1], current_info[2],
                                                       current_info[3], current_info[4])
            if key is None and one_dataset is None:
                continue
            else:
                dataset[key_with_para] = one_dataset
    with open(training_date_file, 'w') as f:
        json.dump(dataset, f)
    f.close()
    return dataset
