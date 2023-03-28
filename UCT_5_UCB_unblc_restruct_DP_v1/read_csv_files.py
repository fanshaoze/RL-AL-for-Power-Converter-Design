import csv
import os
from copy import deepcopy
from datetime import datetime
def main(folder):
    for root, dirs, files in os.walk(folder):
        print(files)
    rows = []
    for file_name in files:
        with open(folder+'/'+file_name, 'r') as f:
            reader = csv.reader(f)

            j = 0
            idx = 0
            for row in reader:
                if j == 0:
                    j += 1
                    # idx += 1
                    header = deepcopy(row)
                    continue
                rows.append(row)
    rows.insert(0, header)
    date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    simu_out_file_name = folder+"/DP-" + str(date_str) + ".csv"
    with open(simu_out_file_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(rows)
    f.close()
    return
if __name__ == '__main__':
    # read_result('mutitest_50-2021-04-17-16-29-41-37526.txt')
    main('DP-result-8-1/simu')
