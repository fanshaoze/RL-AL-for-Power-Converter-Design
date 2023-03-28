import numpy
import json
import argparse
import csv
def preprocess_data(tsv_file_path):
    """load tsv_file and save image1_path and image2_path and caption in json and save it.

    Args:
    tsv_file_path

    Returns:
    None
    """
    split_file = json.load(open(tsv_file_path + "topo_split.json", 'r'))
    data = json.load(open(tsv_file_path + "4comp_new.json", 'r'))
    training_list = split_file["training"]
    testing_list = split_file["test"]

    data_train = []
    data_test = []
    for name in training_list:
        cur = data[name]
        data_train.append({"paths":cur["paths"], "eff":cur["eff"], "vout":cur["vout"], "name":name})

    for name in testing_list:
        cur = data[name]
        data_test.append({"paths": cur["paths"], "eff":cur["eff"], "vout":cur["vout"], "name":name})
        
    n = int(0.9 * len(data_train))
    data_train = data_train[:n]

    with open('./data_train_90.json', 'w') as outfile:
        json.dump(data_train, outfile)

    with open('./data_test.json', 'w') as outfile:
        json.dump(data_test, outfile)




def main(args):
    ''' Main function '''

    preprocess_data(args.tsv_file_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-tsv_file_path', required=True)
    args = parser.parse_args()
    main(args)
