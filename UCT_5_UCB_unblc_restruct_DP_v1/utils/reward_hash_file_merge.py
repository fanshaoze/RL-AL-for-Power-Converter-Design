def whole_merge(file_list, merged_file_name):
    graph_2_reward = {}
    for file in file_list:
        fo_conf = open(file, "r")
        # line = fo_conf.readline()
        while True:
            line = fo_conf.readline()
            # print(line)
            if not line:
                break
            key_value = line.split('$')
            topo = key_value[0]
            str_list = key_value[1]
            if not graph_2_reward.__contains__(topo):
                graph_2_reward[topo] = str_list
            else:
                print(topo)
        print(len(graph_2_reward))
        fo_conf.close()
    fo_conf = open(merged_file_name, "w")
    for (k, v) in graph_2_reward.items():
        fo_conf.write(k + '$' + v)
    fo_conf.close()


def increasing_merge(file_list):
    file_result = "reward_hash.txt"
    fo_conf_result = open(file_result, "w")
    graph_2_reward = {}
    for file in file_list:
        fo_conf = open(file, "r")
        line = fo_conf.readline()
        start_add_line = int(line)
        for _ in range(start_add_line):
            line = fo_conf.readline()
        while True:
            line = fo_conf.readline()
            if not line:
                break
            key_value = line.split('#')
            print(key_value)
            topo = key_value[0]

            str_list = key_value[1]
            if str_list.find('\n') == -1:
                str_list += '\n'
            if not graph_2_reward.__contains__(topo):
                graph_2_reward[topo] = str_list
        fo_conf.close()
    fo_conf.close()

    fo_conf_result.write(str(start_add_line + len(graph_2_reward)) + '\n')

    fo_conf = open(file_list[0], "r")
    line = fo_conf.readline()
    start_add_line = int(line)
    for _ in range(start_add_line):
        line = fo_conf.readline()
        fo_conf_result.write(line)
    for (k, v) in graph_2_reward.items():
        fo_conf_result.write(k + '#' + v)
    fo_conf.close()

    fo_conf_result.close()


# topo_info_hash_0

# file_list = ['tmp_hash/encode_graph_info_hash_1.txt', 'tmp_hash/encode_graph_info_hash_0.txt']
# merged_file_name = 'tmp_hash/encode_graph_info_hash.txt'
# whole_merge(file_list, merged_file_name)

# file_list = ['tmp_hash/topo_info_hash_0.txt',
#              'tmp_hash/topo_info_hash_1.txt',
#              'tmp_hash/topo_info_hash_2.txt',
#              'tmp_hash/topo_info_hash_3.txt']
# merged_file_name = 'tmp_hash/topo_info_hash.txt'
# whole_merge(file_list, merged_file_name)

file_list = ['tmp_hash/encode_topo_info_hash_0.txt',
             'tmp_hash/encode_topo_info_hash_1.txt',
             'tmp_hash/encode_topo_info_hash_2.txt',
             'tmp_hash/encode_topo_info_hash_3.txt']
merged_file_name = 'tmp_hash/encode_topo_info_hash.txt'
whole_merge(file_list, merged_file_name)
