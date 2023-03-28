import json

from topo_analysis.data_2_GCN import parse_data_on_topology
from topo_analysis.topoGraph import TopoGraph
from topo_analysis.parseEfficiencyVOut import parse_eff_and_vout_from_csv
from topo_utils.plot import plot_hist

base_dir = '5comp/'
data_json = base_dir + 'data.json'
result_csv = base_dir + 'sim.csv'
output = '5comp_sim'


if __name__ == '__main__':
    # generate matrices file
    topos = parse_data_on_topology(data_json)

    # parse eff, vout
    effs, vouts = parse_eff_and_vout_from_csv(result_csv)

    # plot distribution when duty cycle = 0.6
    plot_hist(effs.values(), 'efficiency', 'efficiency_dist', bins=50)
    plot_hist(vouts.values(), 'vout', 'vout_dist', bins=50)

    # using eff keys. if a graph is invalid, it's not in eff.keys()
    data = []
    for name in effs.keys():
        node_list = topos[name]['node_list']
        edge_list = topos[name]['edge_list']

        eff = effs[name]
        vout = vouts[name]

        paths = TopoGraph(node_list=node_list, edge_list=edge_list).find_end_points_paths_as_str()
        # this data format can be used for transformer training
        data.append({'node_list': node_list,
                     'edge_list': edge_list,
                     'eff': eff,
                     'vout': vout,
                     'paths': paths,
                     'name': name})

    # generate json containing path info
    with open(output + '.json', 'w') as f:
        json.dump(data, f)

    # generate json of path frequency
    #freqAnalysis.process(output)
