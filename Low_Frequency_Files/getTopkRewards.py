import csv
import json
import os


def main(output_folder, k_list, seed_length=None):
    traj_header = []
    output_for_ks = {}
    for root, dirs, files in os.walk(output_folder):
        print(files)
    traj_topks = json.load(open(output_folder + '/' + files[0]))
    if seed_length is None:
        for traj, seed_length in traj_topks.items():
            seed_length = traj_topks[traj]
            traj_header.append('traj=' + str(traj))
    for k in k_list:
        output_for_ks[k] = []
        file_idx = 0
        for file_name in files:
            for i in range(len(seed_length)):
                output_for_ks[k].append([])
            traj_topks = json.load(open(output_folder + '/' + file_name))
            for traj, seed_topks in traj_topks.items():
                for i in range(len(seed_topks)):
                    topks = seed_topks[i]
                    reward = max(topks[-k:])
                    output_for_ks[k][i+file_idx*len(seed_length)].append(reward)
            file_idx += 1

    print(output_for_ks)
    for k in k_list:
        with open(output_folder + '/diff_topks-' + str(k) + '-results' + '.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(traj_header)
            csv_writer.writerows(output_for_ks[k])
        f.close()


if __name__ == '__main__':
    k_list = [1, 2, 3, 100]
    # file_name = 'BF-50-save-topks-2021-11-17-11-12-51-213426'
    output_folder = './Results/TopkTest'
    main(output_folder=output_folder, k_list=k_list)
