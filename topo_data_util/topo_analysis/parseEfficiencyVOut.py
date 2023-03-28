import collections
import csv
import json


def parse_eff_and_vout_from_txt(filename):
    """
    Parse efficiency and v_out data in analysis_result.txt
    and save them in EFF_OUTPUT, VOUT_OUTPUT, respectively.

    Output formats:
    {'PCC-xxxxxx': efficiency, ...}
    {'PCC-xxxxxx': vout, ...}
    """
    file = open(filename, 'r')

    expect_eff = False
    expect_vout = False

    eff_dict = {}
    vout_dict = {}

    for line in file:
        if line.startswith('PCC'):
            id = line.strip()
            expect_eff = True
        elif expect_eff:
            if line.startswith('efficiency'):
                eff_str = line.strip().split(':')[1]
                # get rid of % symbol
                eff = int(eff_str.split('%')[0])
                eff_dict[id] = .01 * eff

                expect_vout = True

            expect_eff = False
        elif expect_vout:
            if line.startswith('output voltage'):
                vout_str = line.strip().split(':')[1]
                vout = int(vout_str)
                vout_dict[id] = vout

            expect_vout = False

    return eff_dict, vout_dict


def parse_eff_and_vout_from_csv(filename):
    eff_dict = collections.defaultdict(dict)
    vout_dict = collections.defaultdict(dict)

    with open(filename) as f:
        data = csv.reader(f, delimiter=',')
        for row in data:
            name = row[0].strip('\'')

            try:
                duty = float(row[1])
                vout = int(row[2])
                eff = .01 * int(row[3])

                if duty == 0.6:
                    eff_dict[name] = eff
                    vout_dict[name] = vout
            except ValueError:
                #print('invalid topo', row)
                pass

    return eff_dict, vout_dict