# topology construction: 
# 1. select k components
# 2. construct an initial set of join points, consisting of 2 * k points for the left and right join point for the components, and three special join points (VIN, VOUT, GND) [joint point array]
# 3. for each current point p1 in the (2*k + 3) points:
#         select another point p2 from the [joint point array], 
#                       satisfying  (1) it is not a point of the same component or its union set (i.e., the left and right of the same device), 
#                                               (2) if the current point p1 is VIN, VOUT, or GND, then p2 cannot be the VIN, VOUT or GND or its union set, and 
#                                               (3) p1 and p2 cannot be already in the same union set.
#         union the selected point (p1) and the current point (p2) in the join point array
# new valid condition:
#     (1) there is a path from VIN to VOUT, and (2) there is a path from VIN to GND
import collections
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import random
import time
import json
import os

# Added, as the shutil report error in windows
import shutil


def union(x, y, parent):
	f_x = find(x, parent)
	f_y = find(y, parent)

	if f_x == f_y:
		return False

	parent[f_x] = f_y

	return True


def find(x, parent):
	if parent[x] != x:
		parent[x] = find(parent[x], parent)

	return parent[x]


def initial(n):
	# number of basic component in topology

	component_pool = ["GND", 'VIN', 'VOUT']

	port_pool = ["GND", 'VIN', 'VOUT']

	basic_compoments = ["FET-A", "FET-B", "capacitor", "inductor"]

	count_map = {"FET-A": 0, "FET-B": 0, "capacitor": 0, "inductor": 0}

	comp2port_mapping = {0: [0], 1: [1], 2: [2]}  # key is the idx in component pool, value is idx in port pool

	port2comp_mapping = {0: 0, 1: 1, 2: 2}

	index = range(4)

	port_2_idx = {"GND": 0, 'VIN': 1, 'VOUT': 2}

	idx_2_port = {0: 'GND', 1: 'VIN', 2: "VOUT"}

	same_device_mapping = {}

	graph = collections.defaultdict(set)

	for i in range(n):
		idx = random.choice(index)

		count = str(count_map[basic_compoments[idx]])

		count_map[basic_compoments[idx]] += 1

		component = basic_compoments[idx] + '-' + count

		component_pool.append(component)

		idx_component_in_pool = len(component_pool) - 1

		port_pool.append(component + '-left')

		port_pool.append(component + '-right')

		port_2_idx[component + '-left'] = len(port_2_idx)

		port_2_idx[component + '-right'] = len(port_2_idx)

		comp2port_mapping[idx_component_in_pool] = [port_2_idx[component + '-left'], port_2_idx[component + '-right']]

		port2comp_mapping[port_2_idx[component + '-left']] = idx_component_in_pool

		port2comp_mapping[port_2_idx[component + '-right']] = idx_component_in_pool

		idx_2_port[len(idx_2_port)] = component + '-left'

		idx_2_port[len(idx_2_port)] = component + '-right'

		same_device_mapping[port_2_idx[component + '-left']] = port_2_idx[component + '-right']

		same_device_mapping[port_2_idx[component + '-right']] = port_2_idx[component + '-left']

		# parent = [-1] * len(port_pool)
	parent = list(range(len(port_pool)))

	# print("port_2_idx",port_2_idx)
	# print("idx_2_port",idx_2_port)
	# print("port_pool",port_pool)
	# print("component_pool",component_pool)
	# print("same_device_mapping",same_device_mapping)
	# print("comp2port_mapping",comp2port_mapping)
	# print("port2comp_mapping",port2comp_mapping)

	return component_pool, port_pool, count_map, comp2port_mapping, port2comp_mapping, port_2_idx, idx_2_port, same_device_mapping, graph, parent


def convert_to_netlist(graph, component_pool, port_pool, parent, comp2port_mapping):
	# for one component, find the two port
	# if one port is GND/VIN/VOUT, leave it don't need to find the root
	# if one port is normal port, then find the root, if the port equal to root then leave it, if the port is not same as root, change the port to root
	list_of_node = set()

	list_of_edge = set()

	# netlist = []

	for idx, comp in enumerate(component_pool):
		# cur = []
		# cur.append(comp)
		list_of_node.add(comp)
		for port in comp2port_mapping[idx]:
			port_joint_set_root = find(port, parent)
			# cur.append(port_pool[port_joint_set_root])
			if port_joint_set_root in [0, 1, 2]:
				list_of_node.add(port_pool[port_joint_set_root])
				# list_of_edge.add((comp, port_pool[port_root]))
				list_of_node.add(port_joint_set_root)
				list_of_edge.add((comp, port_joint_set_root))
				list_of_edge.add((port_pool[port_joint_set_root], port_joint_set_root))
			else:
				list_of_node.add(port_joint_set_root)
				list_of_edge.add((comp, port_joint_set_root))

	netlist = []
	joint_list = set()

	for idx, comp in enumerate(component_pool):
		if comp in ['VIN', 'VOUT', 'GND']:
			continue
		cur = []

		cur.append(comp)
		for port in comp2port_mapping[idx]:
			# print(port_joint_set_root)
			port_joint_set_root = find(port, parent)
			root_0 = find(0, parent)
			root_1 = find(1, parent)
			root_2 = find(2, parent)
			if port_joint_set_root == root_0:
				cur.append("0")
			elif port_joint_set_root == root_1:
				cur.append("IN")

			elif port_joint_set_root == root_2:
				cur.append("OUT")
				# cur.append(port_pool[port_joint_set_root])
			# else:
			else:
				joint_list.add(str(port_joint_set_root))
				cur.append(str(port_joint_set_root))
		netlist.append(cur)

	return list(list_of_node), list(list_of_edge), netlist, list(joint_list)


def convert_graph(graph, comp2port_mapping, port2comp_mapping, idx_2_port, parent, component_pool, same_device_mapping,
                  port_pool):
	list_of_node = set()

	list_of_edge = set()

	has_short_cut = False

	for node in comp2port_mapping:
		if len(comp2port_mapping[node]) == 2:
			list_of_node.add(comp2port_mapping[node][0])
			list_of_node.add(comp2port_mapping[node][1])
			list_of_edge.add((comp2port_mapping[node][1], comp2port_mapping[node][0]))

	for node in graph:
		root_node = find(node, parent)
		list_of_node.add(node)
		list_of_node.add(root_node)

		# cur_node_the_other_port_root = find(cur_node_the_other_port, parent)

		if node in same_device_mapping:
			cur_node_the_other_port = same_device_mapping[node]
			cur_node_the_other_port_root = find(cur_node_the_other_port, parent)
			if cur_node_the_other_port_root == root_node:
				has_short_cut = True

		if root_node != node:
			list_of_edge.add((node, root_node))

		for nei in graph[node]:
			list_of_node.add(nei)
			if nei != root_node:
				list_of_edge.add((nei, root_node))

	return list(list_of_node), list(list_of_edge), has_short_cut


def save_ngspice_file(netlist, path, args):
	# file_name = "PCC-" + format(n, '06d') + '.cki'
	path += '.cki'
	file = open(path, 'w')
	print(path)
	# capacitor_paras = [0.1 , 0.2, 0.47, 0.56, 0.68, 1, 2.2, 4.7, 5.6, 6.8, 10, 22, 47, 56]#uF
	# inductor_paras =  [0.1, 0.2, 0.4, 0.6, 0.8, 1, 10, 20, 40, 60, 80, 100]#uH

	capacitor_paras = args.cap_paras
	inductor_paras = args.ind_paras
	selected_cap = random.choice(capacitor_paras)
	# selected_cap = 10
	selected_ind = random.choice(inductor_paras)
	# selected_ind = 100

	prefix = [
		".title buck.cki",
		".model MOSN NMOS level=8 version=3.3.0",
		".model MOSP PMOS level=8 version=3.3.0",
		".PARAM " + args.ngspice_para + " ind=%su cap=%su" % (selected_ind, selected_cap),
		"",
		".SUBCKT DC_source 1",
		"VIN 2 0 {vin}",
		"ROUT 2 1 {rvinout}",
		".ENDS",
		"",
		".SUBCKT CAP_res 1 2 ",
		"C1 1 2 {cap}",
		"R1 1 2 {rcap}",
		".ENDS",
		"",
		".SUBCKT IND_res 1 2",
		"L1 1 3 {ind}",
		"R1 3 2 {rind}",
		".ENDS",
		"",
		"\n",
		"*input*",
		"Vclock1 gaten 0 PULSE (0 {vin}  {abs(2*D-1)/2/freq} 1n 1n {D/freq} {1/freq})",
		"Vclock2 gatep 0 PULSE (0 {vin} 0 1n 1n {(1-D)/freq} {1/freq})",
		# "Rgate_n gaten_r gaten 0.001",
		# "Rgate_p gatep_r gatep 0.001",
		"XVIN IN_ext DC_source",
		"RINsense IN_ext IN 0.001",
		"ROUTsense OUT OUT_ext 0.001",
		"\n"]

	sufix = ["\n",
	         "COUT OUT 0 10u",
	         "RLOAD OUT_ext 0 {rload}",
	         ".save all",
             # ".save i(vind)",
	         ".control",
	         "tran 0.1u 4000u",
	         "print V(OUT)",
             # ".print tran V(IN_ext,IN)",
             # ".print tran V(OUT,OUT_ext)",
	         "print V(IN_ext,IN)",
	         "print V(OUT,OUT_ext)",
	         ".endc",
	         ".end",
             ]

	file.write("\n".join(prefix) + '\n')
	file.write("*topology*" + '\n')
	for x in netlist:
		# print(x)÷
		x[0] = x[0].replace("-", "")
		if "capacitor" in x[0]:
			x[0] = x[0].replace("capacitor", "XC")
			x.append("CAP_res")
		elif "inductor" in x[0]:
			x[0] = x[0].replace("inductor", "XL")
			x.append("IND_res")
		elif "FETA" in x[0]:  # nmos
			x[0] = "M" + x[0]
			x.insert(2, "gaten")
			x.append("0 MOSN L=1U W=2000U")
		elif "FETB" in x[0]:
			x[0] = "M" + x[0]
			x.insert(2, "gatep")
			x.append("IN MOSP L=1U W=5000U")
	file.write("\n".join([' '.join(x) for x in netlist]) + '\n')
	file.write("\n".join(sufix))
	file.close()
	return


def save_ngspice_file_without_args(netlist, path, capacitor_paras, inductor_paras, ngspice_para):
	# file_name = "PCC-" + format(n, '06d') + '.cki'
	path += '.cki'
	file = open(path, 'w')
	print(path)
	# capacitor_paras = [0.1 , 0.2, 0.47, 0.56, 0.68, 1, 2.2, 4.7, 5.6, 6.8, 10, 22, 47, 56]#uF
	# inductor_paras =  [0.1, 0.2, 0.4, 0.6, 0.8, 1, 10, 20, 40, 60, 80, 100]#uH

	selected_cap = random.choice(capacitor_paras)
	# selected_cap = 10
	selected_ind = random.choice(inductor_paras)
	# selected_ind = 100

	prefix = [
		".title buck.cki",
		".model MOSN NMOS level=8 version=3.3.0",
		".model MOSP PMOS level=8 version=3.3.0",
		".PARAM " + ngspice_para + " ind=%su cap=%su" % (selected_ind, selected_cap),
		"",
		".SUBCKT DC_source 1",
		"VIN 2 0 {vin}",
		"ROUT 2 1 {rvinout}",
		".ENDS",
		"",
		".SUBCKT CAP_res 1 2 ",
		"C1 1 2 {cap}",
		"R1 1 2 {rcap}",
		".ENDS",
		"",
		".SUBCKT IND_res 1 2",
		"L1 1 3 {ind}",
		"R1 3 2 {rind}",
		".ENDS",
		"",
		"\n",
		"*input*",
		"Vclock1 gaten 0 PULSE (0 {vin}  {abs(2*D-1)/2/freq} 1n 1n {D/freq} {1/freq})",
		"Vclock2 gatep 0 PULSE (0 {vin} 0 1n 1n {(1-D)/freq} {1/freq})",
		# "Rgate_n gaten_r gaten 0.001",
		# "Rgate_p gatep_r gatep 0.001",
		"XVIN IN_ext DC_source",
		"RINsense IN_ext IN 0.001",
		"ROUTsense OUT OUT_ext 0.001",
		"\n"]

	sufix = ["\n",
	         "COUT OUT 0 10u",
	         "RLOAD OUT_ext 0 {rload}",
	         ".save all",
             # ".save i(vind)",
	         ".control",
	         "tran 0.1u 4000u",
	         "print V(OUT)",
             # ".print tran V(IN_ext,IN)",
             # ".print tran V(OUT,OUT_ext)",
	         "print V(IN_ext,IN)",
	         "print V(OUT,OUT_ext)",
	         ".endc",
	         ".end",
             ]

	file.write("\n".join(prefix) + '\n')
	file.write("*topology*" + '\n')
	for x in netlist:
		# print(x)÷
		x[0] = x[0].replace("-", "")
		if "capacitor" in x[0]:
			x[0] = x[0].replace("capacitor", "XC")
			x.append("CAP_res")
		elif "inductor" in x[0]:
			x[0] = x[0].replace("inductor", "XL")
			x.append("IND_res")
		elif "FETA" in x[0]:  # nmos
			x[0] = "M" + x[0]
			x.insert(2, "gaten")
			x.append("0 MOSN L=1U W=2000U")
		elif "FETB" in x[0]:
			x[0] = "M" + x[0]
			x.insert(2, "gatep")
			x.append("IN MOSP L=1U W=5000U")
	file.write("\n".join([' '.join(x) for x in netlist]) + '\n')
	file.write("\n".join(sufix))
	file.close()
	return


def nets_to_ngspice_files(topologies, configs, n_components=4,
                          output=False, output_folder="component_data_random"):
	ngspice_para = "\"freq=%s vin=%s rload=50 iout=0.2 rcap=1Meg rind=0.1 rvinout=0.1 D=%s\"" % \
	               (configs["freq"], configs["vin"], configs["D"])
	cap_paras = "1, 10"  # or "10,20"
	ind_paras = "1, 10"  # or ,"30,50"
	n_topology = len(topologies)
	sys_os = configs['sys_os']

	cap_paras = [float(x) for x in cap_paras.split(',')]
	ind_paras = [float(x) for x in ind_paras.split(',')]
	directory_path = str(n_components) + output_folder

	if sys_os == "windows":
		# windows command delete prior topo_analysis
		os.system("del /Q %s\*" % directory_path)

	if not os.path.exists(directory_path):
		os.mkdir(directory_path)

	for filename in os.listdir(directory_path):
		file_path = os.path.join(directory_path, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))

	for i in range(0, len(topologies)):
		port_2_idx = topologies[i].port_2_idx
		idx_2_port = topologies[i].idx_2_port
		component_pool = topologies[i].component_pool
		same_device_mapping = topologies[i].same_device_mapping
		port_pool = topologies[i].port_pool
		component_pool = topologies[i].component_pool
		parent = topologies[i].parent
		comp2port_mapping = topologies[i].comp2port_mapping
		port2comp_mapping = {}
		for key in comp2port_mapping.keys():
			for v in comp2port_mapping[key]:
				port2comp_mapping[v] = key
		data = {}
		list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(None, component_pool,
		                                                                     port_pool, parent, comp2port_mapping)

		T = nx.Graph()
		T.add_nodes_from(list_of_node)
		T.add_edges_from(list_of_edge)
		# plt.figure(1)
		# nx.draw(G, with_labels=True)
		plt.figure()
		nx.draw(T, with_labels=True)

		name = "PCC-" + format(i, '06d')
		file_path = directory_path + '/' + name
		# print(graph_name)
		plt.savefig(file_path)  # save as png
		if output:
			plt.show()
		T.clear()
		plt.close()

		data[name] = {
			"port_2_idx": port_2_idx,
			"idx_2_port": idx_2_port,
			"port_pool": port_pool,
			"component_pool": component_pool,
			"same_device_mapping": same_device_mapping,
			"comp2port_mapping": comp2port_mapping,
			"port2comp_mapping": port2comp_mapping,
			"list_of_node": list_of_node,
			"list_of_edge": list_of_edge,
			"netlist": netlist,
			"joint_list": joint_list,
		}
		save_ngspice_file_without_args(netlist, file_path, cap_paras, ind_paras, ngspice_para)

		with open(directory_path + '/data.json', 'w') as outfile:
			json.dump(data, outfile)
