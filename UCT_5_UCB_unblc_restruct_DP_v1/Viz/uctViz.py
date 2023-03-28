import os
import sys
import math
import random
import shutil

import graphviz
from graphviz import Digraph
from ucts import uct
import datetime
from ucts import TopoPlanner


def delAllFiles(rootdir):
    filelist = []
    filelist = os.listdir(rootdir)  # list all the files
    for f in filelist:
        filepath = os.path.join(rootdir, f)  # path-absolute path
        if os.path.isfile(filepath):  # is file?
            os.remove(filepath)  # delete file
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)  # is dir


class TreeGraph(object):

    def __init__(self, node_list_):
        self.grap_g = Digraph("G", format="png")
        self.sub_g0 = Digraph(comment="process1", graph_attr={"style": 'filled'})

        self.node_index = -1
        self.node_list = node_list_
        self.dest_leaf_index = 3
        # self.dest_leaf_index = len(self.node_list)-1
        self.path_index = []

    # def generate_label(self, Index, state):
    # 	# need modified for circulate
    # 	return "(" + str(state.x) + "," + str(state.y) + "," + str(state.food) + ") #" + str(Index)

    def generate_label_com(self, index, state):
        # need modified for circulate
        state_label = str(state.component_pool)
        state_label = state_label.replace("inductor","ind")
        state_label = state_label.replace("capacitor","cap")
        state_label = state_label.replace("FET-A","FA")
        state_label = state_label.replace("FET-B","FB")
        state_label = state_label.replace(']', '')
        state_label = state_label.replace('[', '')
        state_label = state_label.replace(' ', '')
        return state_label

    def generate_label(self, index, state):
        # need modified for circulate
        state_label = str(state.graph)[26:]
        state_label = state_label.replace(':', '')
        state_label = state_label.replace('(', '')
        state_label = state_label.replace(')', '')
        state_label = state_label.replace('{', '')
        return str(index)+"#"+state_label

    def generate_terminal_label(self, index, state):
        # need modified for circulate
        if state.step - (len(state.component_pool) - 3) >= len(state.port_pool):
            return "T on #" + str(index)
        else:
            return "F on #" + str(index)

    def generate_step_label(self, state):
        # need modified for circulate
        return "(" + str(state.step) + ")"

    def generate_action_label(self, action):
        return str(action.value)

    def generate_state_value_label(self, state_node):
        # need modified for circulate
        return "(" + str('%.7f' % state_node.first_MC_) + "),"
    def generate_state_reward_label(self, state_node):
        # need modified for circulate
        return "(r" + str('%.7f' % state_node.reward_) + "),"

    # uct_tree is the Tree,
    def generate_action_value_label(self, action_node):
        return str(action_node.num_visits_) + "," + str('%.7f' % action_node.avg_return_)

    def get_node_index(self, state_node):
        for i in range(0, len(self.node_list)):
            if state_node.state_.equal(self.node_list[i].state_):
                return i
        return -1

    def path_search(self, state_node):
        path_index = []
        path_index.append(self.get_node_index(state_node))
        while state_node.parent_act_ is not None:
            parent_act = state_node.parent_act_
            state_node = parent_act.parent_state_
            path_index.append(self.get_node_index(state_node))
        return path_index

    def drawAll(self, uct_tree, folder, save_final=False):
        if save_final:
            self.grap_g = None
            self.sub_g0 = None
            self.grap_g = Digraph("G", format="png")
            self.sub_g0 = Digraph(comment="process1", graph_attr={"style": 'filled'})
            file_name = folder + "/test-table"
            self.draw_search(uct_tree, len(self.node_list) - 1, file_name)
            return
        for i in range(0, len(self.node_list)):
            self.grap_g = None
            self.sub_g0 = None
            self.grap_g = Digraph("G", format="png")
            self.sub_g0 = Digraph(comment="process1", graph_attr={"style": 'filled'})
            file_name = folder + "/test-table" + str(i)
            self.draw_search(uct_tree, i, file_name)
        return

    def draw_search(self, uct_tree, _dest_leaf_index, file_name):
        self.dest_leaf_index = _dest_leaf_index
        # self.path_index = self.path_search(self.node_list[self.dest_leaf_index])
        # print(self.path_index)
        self.node_index += 1
        root_index = self.node_index
        root_index = self.get_node_index(uct_tree.root_)
        self.sub_g0.node(str(root_index), self.generate_label(root_index, uct_tree.root_.state_),
                         _attributes={"style": "filled", "color": "grey"})
        self.drawTree(root_index, uct_tree.root_, None, None, None)
        self.grap_g.subgraph(self.sub_g0)
        self.grap_g.render(file_name, view=False)

    # def get_line_label(self, parent_state, aid):
    # 	if aid == 0:
    # 		if (parent_state.state_.x == 0) and (parent_state.state_.y > 0):
    # 			return "down"
    # 	elif aid == 1:
    # 		if (parent_state.state_.x == 0) and (parent_state.state_.y < 4):
    # 			return "up"
    # 	elif aid == 2:
    # 		if parent_state.state_.x > 0:
    # 			return "left"
    # 	elif aid == 3:
    # 		if parent_state.state_.x < 4:
    # 			return "right"
    # 	return "stay"

    def get_line_label(self, action_value):
        return str(action_value)

    def drawTree(self, current_index, v_state_node, parents_index, parent_state, action_value):
        actVisitCounter = 0
        act_size = len(v_state_node.node_vect_)
        # print("1##",self.generate_label(current_index,v_state_node.state_))
        if parent_state is not None:
            # print("2##",self.generate_label(parents_index,parent_state.state_))
            line_label = self.get_line_label(action_value)

            # if current_index > self.dest_leaf_index:
            #     self.grap_g.edge(str(parents_index), str(current_index), line_label)
            # elif current_index in self.path_index:
            #     self.grap_g.edge(str(parents_index), str(current_index), line_label)
            # else:
            #     self.grap_g.edge(str(parents_index), str(current_index))

            self.grap_g.edge(str(parents_index), str(current_index), line_label)

        for i in range(0, act_size):
            action_value = v_state_node.act_vect_[i].value
            # print (action_value)
            v_action_node = v_state_node.node_vect_[i]
            state_size = len(v_action_node.state_vect_)
            for j in range(0, state_size):
                v_childstate_node = v_action_node.state_vect_[j]
                # print("3##",self.generate_label(action_value,v_childstate_node.state_))
                self.node_index += 1
                child_index = self.get_node_index(v_childstate_node)
                # if child_index in self.path_index:
                # 	self.sub_g0.node(str(child_index), self.generate_label(child_index, v_childstate_node.state_) \
                # 					 , _attributes={"style": "filled", "color": "grey"})
                # elif child_index < self.dest_leaf_index:
                # 	self.sub_g0.node(str(child_index), self.generate_label(child_index, v_childstate_node.state_))
                # else:
                # 	self.sub_g0.node(str(child_index), self.generate_label(child_index, v_childstate_node.state_) \
                # 					 , _attributes={"style": "filled", "color": "white"}, fontcolor="white")

                self.sub_g0.node(str(child_index), self.generate_label_com(child_index, v_childstate_node.state_))

                # self.sub_g0.node(str(child_index), self.generate_label(child_index, v_childstate_node.state_) + "\n" +
                #                  self.generate_state_value_label(v_childstate_node) +
                #                  self.generate_state_reward_label(v_childstate_node) +
                #                  self.generate_action_value_label(v_action_node))
                # self.sub_g0.node(str(child_index), self.generate_terminal_label(child_index, v_childstate_node.state_) +
                #                  "\n" + self.generate_action_value_label(0, v_action_node))
                # print("4##",self.generate_label(child_index,v_childstate_node.state_))
                # print("5##",self.generate_label(current_index,v_state_node.state_))
                self.drawTree(child_index, v_childstate_node, current_index, v_state_node, action_value)

    def drawExpend(self, sim):
        pass

#
# depthList = [4]
# trajectory = [10]
# initPositionList = [(2, 3, 1)]
# numGames = 1
# outfile_name = "multitestpro-out.txt"
# starttime = datetime.datetime.now()
# fo = open(outfile_name, "w")
# fo.write("maxdepth,num_Runs,avgstep\n")
# sim = uctPlanner.ToySimulator()
# sim2 = uctPlanner.ToySimulator()
# avgsteps = 0
# avgstep_list = []
# initNums = len(initPositionList)
# initsize = 0
# ransize = 0
# # (self, _sim, _maxDepth, _numRuns, _ucbScalar, _gamma, _leafValue, _endEpisodeValue):
# for max_depth in depthList:
# 	for num_Runs in trajectory:
# 		print(max_depth, ",", num_Runs)
# 		avgsteps = 0
# 		uct_tree = uct.UCTPlanner(sim2, max_depth, num_Runs, 1, 0.95, 0, 0)
# 		# uct_tree = uct.UCTPlanner(sim2, 30, 110, 1, 0.95, 0, 0)
# 		print(numGames, initNums)
# 		for j in range(0, initNums):
# 			initstate = uctPlanner.ToyState(initPositionList[j][0], initPositionList[j][1], initPositionList[j][2])
# 			for i in range(0, int(numGames / initNums)):
# 				sim.setState(initstate)
# 				steps = 0
# 				r = 0
# 				# sim.getState().print()
# 				while not sim.isTerminal():
# 					steps += 1
# 					uct_tree.setRootNode(sim.getState(), sim.getActions(), r, sim.isTerminal())
# 					# print()
# 					node_list = uct_tree.plan()
# 					# viz start
# 					folder = "./Results/viz" + str(steps)
# 					isExists = os.path.exists(folder)
# 					if not isExists:
# 						os.makedirs(folder)
# 					else:
# 						delAllFiles(folder)
# 					treeviz = TreeGraph(node_list)
# 					treeviz.drawAll(uct_tree, folder)
#
# 					# viz finished
# 					# return the action with the highest reward
# 					action = uct_tree.getAction()
# 					# print("-action:", end='')
# 					# action.print()
# 					# print("->", end='')
# 					# uct_tree.testTreeStructure()
# 					# 先序遍历，测试所有节点
# 					# uct_tree.testDeterministicProperty()
# 					r = sim.act(action)
# 					# sim.getState().print()
# 					# print("reward:",uct_tree.root_.reward_,end='')
# 					# print("")
# 					# sim.getState().print()
# 					break
# 				# sim = sim.reset()
# 				print("#####################Game:", i, "  steps: ", steps, "  r: ", r)
# 				avgsteps += steps
# 		avgsteps = avgsteps / numGames
# 		fo.write(str(max_depth) + "," + str(num_Runs) + "," + str(avgsteps) + "\n")
# 		print(max_depth, ",", num_Runs, ":", avgsteps)
#
# 		avgstep_list.append(avgsteps)
# endtime = datetime.datetime.now()
# print("execute time: ", (endtime - starttime).seconds)
# # avgtimes+=(endtime - starttime).seconds
# fo.close()
