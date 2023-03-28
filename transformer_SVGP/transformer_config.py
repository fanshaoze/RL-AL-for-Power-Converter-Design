# global config for transformer
# these are not likely to be changed

# the maximum number of components in a path
# need to update this if consider larger topologies
max_path_len = 8
# the maximum number of paths in a topology
# paths exceeding this limit are dropped
max_path_num = 16

# lstm architecture
bidirectional = True
average_pooling = True
