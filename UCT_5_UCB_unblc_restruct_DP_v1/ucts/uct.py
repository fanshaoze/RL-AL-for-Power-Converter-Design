"""The UCT model.

This file include the part of UCT, including the definitions and common methods.
"""

import logging
import sys
import math
import random
import copy
import warnings
import os


class State(object):
    """The definiton of State.

    This is the definiton of State in UCT, the attributes need to be defined
    seperately in different game models
    """

    def equal(self, state):
        """Find out whether two states are equal.

        Args:
          None.

        Returns:
          True if two states are equal.
        """

        pass

    def duplicate(self):
        """Return a deep copy of itself.
        """

        pass

    def print(self):
        pass

    def __del__(self):
        pass


class SimAction(object):
    """The definition of action.

    This is the definition of State in UCT, the attributes need to be defined
    separately in different game models
    """

    def equal(self, act):
        """Find out whether two actions are equal.

        Args:
          None.

        Returns:
          True if two actions are equal.
        """

    pass

    def equivalent(self, act):
        """Find out whether two actions are equal.

        Args:
          None.

        Returns:
          True if two actions are equal.
        """

    pass

    def duplicate(self):
        pass

    def print_state(self):
        pass

    def __del__(self):
        pass


class Simulator(object):
    """state.

    Defined according to the instances
    Longer class information....
    """

    def set_state(self, _next_candidate_components, _current_candidate_components, state):  # Done
        pass

    def get_state(self):
        pass

    def act(self, action):  # equal(SimAction* act) = 0;#Done
        pass

    def get_actions(self):  # Done
        pass

    def get_action_weights(self, branch_num, prestate_comps, prestate_n_can, prestate_c_can,
                           next_comps, next_n_can, nex_c_can):
        pass

    def get_weights(self):
        pass

    def is_terminal(self):  # Done
        pass

    def reset(self):  # Done
        pass


class StateNode(object):

    def __init__(self, _parent_act, _state, _act_vect, _next_candidate_components,
                 _current_candidate_components, _weights, _reward, _is_terminal):  # Done
        self.parent_act_ = _parent_act

        self.state_ = _state.duplicate()
        self.reward_ = _reward
        self.is_terminal_ = _is_terminal
        self.num_visits_ = 0
        self.weighted_num_visits_ = 0
        self.rave_num_visits_ = 0
        self.act_ptr_ = 0
        self.first_MC_ = 0
        self.act_vect_ = []
        self.next_candidate_components_ = _next_candidate_components
        self.current_candidate_components_ = _current_candidate_components
        self.weights_ = _weights
        _size = len(_act_vect)

        for i in range(0, _size):
            self.act_vect_.append(_act_vect[i].duplicate())

        # dictionary
        self.node_vect_ = []  # vector<ActionNode*> node_vect_;

    def __del__(self):  # Done
        del self.state_
        self.act_vect_.clear()

    def is_full(self):  # Done
        return self.act_ptr_ == len(self.act_vect_)

    def add_action_node(self):  # Done
        assert self.act_ptr_ < len(self.act_vect_)
        self.node_vect_.append(ActionNode(self))
        self.act_ptr_ += 1
        return self.act_ptr_ - 1

    def place_action_node(self, branch):  # Done
        assert self.act_ptr_ < len(self.act_vect_)
        self.node_vect_[branch] = ActionNode(self)
        self.act_ptr_ += 1
        return self.act_ptr_ - 1


# #TODO hash function improve
# m = hashlib.md5()
# m.update(world_str.encode('utf-8'))
# return m.hexdigest()


class ActionNode(object):

    def __init__(self, _parent_state):

        self.parent_state_ = _parent_state
        self.avg_return_ = 0
        self.num_visits_ = 0
        self.weighted_num_visits_ = 0
        self.visit_update_weight = 0
        self.rave_avg_return_ = 0
        self.rave_num_visits_ = 0
        self.state_vect_ = []

    def contain_next_state(self, state):
        size = len(self.state_vect_)
        for i in range(0, size):
            if state.equal(self.state_vect_[i].state_):
                return True
        return False

    def get_next_state_node(self, state):
        size = len(self.state_vect_)
        for i in range(0, size):
            if state.equal(self.state_vect_[i].state_):
                return self.state_vect_[i]
        return None

    def add_state_node(self, _state, _act_vect, _next_candidate_components,
                       _current_candidate_components, _weights, _reward, _is_terminal):  # Done
        index = len(self.state_vect_)
        self.state_vect_.append(StateNode(self, _state, _act_vect, _next_candidate_components,
                                          _current_candidate_components, _weights, _reward, _is_terminal))
        return self.state_vect_[index]

    def set_visit_weight(self, _visit_update_weight):  # Done
        self.visit_update_weight = _visit_update_weight

    def place_state_node(self, _state, _act_vect, _next_candidate_components,
                         _current_candidate_components, _weights, _reward, _is_terminal):  # Done
        index = len(self.state_vect_)
        self.state_vect_.append(StateNode(self, _state, _act_vect, _next_candidate_components,
                                          _current_candidate_components, _weights, _reward, _is_terminal))
        for _ in range(len(self.state_vect_[len(self.state_vect_) - 1].act_vect_)):
            self.state_vect_[len(self.state_vect_) - 1].node_vect_.append(None)
        return self.state_vect_[index]

    def __del__(self):  # Done
        pass


class UCTPlanner(object):

    def __init__(self, _sim, _max_depth, _num_runs, _ucb_scalar,
                 _gamma, _leaf_value, _end_episode_value, _deterministic, _rave_scalar, _rave_k,
                 _component_default_policy, _path_default_policy):
        self.sim_ = _sim
        self.max_depth_ = _max_depth
        self.num_runs_ = _num_runs
        self.ucb_scalar_ = _ucb_scalar
        self.rave_scalar_ = _rave_scalar
        self.rave_k_ = _rave_k
        self.gamma_ = _gamma
        self.beta_ = 1
        self.leaf_value_ = _leaf_value
        self.end_episode_value_ = _end_episode_value
        self.deterministic_ = _deterministic
        self.component_default_policy_ = _component_default_policy
        self.path_default_policy_ = _path_default_policy

        self.root_ = None
        self.original_root_ = None

        _leaf_value = 0
        _end_episode_value = 0
        self.node_list = []

    def __del__(self):
        pass

    def get_all_states(self):
        all_states = []
        for node in self.node_list:
            state = node.state_
            if not any(state.equal(s_) for s_ in all_states):
                all_states.append(state)

        return all_states

    def set_root_node(self, _state, _act_vect, _next_candidate_components,
                      _current_candidate_components, _weights, _reward, _is_terminal):
        if self.root_ is not None:
            self.clear_tree()
        self.root_ = StateNode(None, _state, _act_vect, _next_candidate_components,
                               _current_candidate_components, _weights, _reward, _is_terminal)
        # self.root_.node_vect_ = [None for _ in range(len(_act_vect))]
        self.original_root_ = self.root_

    def clear_tree(self):
        self.root_ = None
        pass

    def set_and_plan(self, conn):
        while True:
            action = conn.recv()
            new_root = conn.recv()
            self.sim_.graph_2_reward = conn.recv()
            if action == -1:
                self.set_root_node(new_root.state_, new_root.act_vect_, new_root.next_candidate_components_,
                                   new_root.current_candidate_components_, new_root.weights_, new_root.reward_,
                                   new_root.is_terminal_)
            else:
                self.update_root_node(action, new_root)
            tree_size, tmp_planner, depth = self.plan()
            conn.send(tree_size)
            conn.send(tmp_planner)
            conn.send(depth)

    def add_all_nodes_2_list(self, root, node_list):
        for action_node in root.node_vect_:
            for state_node in action_node.state_vect_:
                node_list.append(state_node)
                self.add_all_nodes_2_list(state_node, node_list)

    def plan(self, step_traj, get_viz=False):
        """
        The MCTS process
        currently the parameters are not used
        """
        if get_viz:
            node_list = [self.root_]  # for viz
            self.add_all_nodes_2_list(self.root_, node_list)
        current = self.root_
        self.sim_.set_state(current.next_candidate_components_, current.current_candidate_components_, current.state_)

        tree_size = 1
        assert self.root_ is not None
        root_offset = 0
        if root_offset == 0:
            self.root_.num_visits_ += 1
            root_offset += 1
        num_runs = step_traj
        print("root offset is: ", root_offset)
        depth = 0
        for trajectory in range(root_offset, num_runs + 1):
            current = self.root_
            mc_return = self.leaf_value_
            depth = 0
            # self.print_nodes_number_of_visits(current)
            while True:
                depth += 1
                self.sim_.set_state(current.next_candidate_components_, current.current_candidate_components_,
                                    current.state_)
                # is current is terminal, we need to stop
                if self.sim_.is_terminal():
                    mc_return = self.sim_.get_reward()
                    break
                elif current.is_full():
                    uct_branch = self.get_UCT_branch_index(current)
                    current = current.node_vect_[uct_branch].state_vect_[0]
                    continue
                else:
                    act_ID = current.add_action_node()
                    self.sim_.set_state(current.next_candidate_components_, current.current_candidate_components_,
                                        current.state_)
                    r = self.sim_.act(current.act_vect_[act_ID])
                    next_node = current.node_vect_[act_ID].add_state_node(self.sim_.get_state(),
                                                                          self.sim_.get_actions(),
                                                                          self.sim_.get_next_candidate_components(),
                                                                          self.sim_.get_current_candidate_components(),
                                                                          self.sim_.get_weights(),
                                                                          r, self.sim_.is_terminal())
                    current.node_vect_[act_ID].set_visit_weight(self.sim_.get_action_weights(len(current.act_vect_),
                                                                                             current.state_.component_pool,
                                                                                             current.next_candidate_components_,
                                                                                             current.current_candidate_components_,
                                                                                             next_node.state_.component_pool,
                                                                                             next_node.next_candidate_components_,
                                                                                             next_node.current_candidate_components_))
                    self.node_list.append(next_node)
                    tree_size += 1
                    if -1 == self.max_depth_:
                        mc_return = self.MC_sampling_terminal(next_node)
                    else:
                        mc_return = self.MC_sampling_depth(next_node, self.max_depth_ - depth)
                    current = next_node
                    if get_viz:
                        node_list.append(current)  # for viz
                    break
            self.update_values(current, mc_return)
        if get_viz:
            return tree_size, self, depth, node_list
        return tree_size, self, depth

    def get_action(self):
        return self.root_.act_vect_[self.get_greedy_branch_index()]

    def get_most_visited_branch_index(self):
        assert self.root_ is not None
        maximizer = []
        size = len(self.root_.node_vect_)
        for i in range(0, size):
            maximizer.append(self.root_.node_vect_[i].num_visits_)
        return maximizer.index(max(maximizer))

    def get_greedy_branch_index(self):
        assert self.root_ is not None
        maximizer = []
        size = len(self.root_.node_vect_)
        for i in range(0, size):
            maximizer.append(self.root_.node_vect_[i].avg_return_)
        if maximizer == []:
            return 0
        else:
            return maximizer.index(max(maximizer))

    def get_UCT_branch_index(self, node):
        """
        Use the UCB to select the next branch
        """
        # det = math.log(float(node.num_visits_))
        det = math.log(float(node.weighted_num_visits_))
        maximizer = []
        size = len(node.node_vect_)
        for i in range(0, size):
            val = node.node_vect_[i].avg_return_
            # val += self.ucb_scalar_ * math.sqrt(det / float(node.node_vect_[i].num_visits_))
            # val = self.ucb_scalar_ * math.sqrt(det / float(node.node_vect_[i].weighted_num_visits_))
            val += self.ucb_scalar_ * math.sqrt(det / float(node.node_vect_[i].weighted_num_visits_))
            maximizer.append(val)
        return maximizer.index(max(maximizer))

    def print_nodes_number_of_visits(self, node):
        size = len(node.node_vect_)
        print("root nodes:", end='')
        for i in range(0, size):
            print(i, ":", node.node_vect_[i].weighted_num_visits_, " ", end='')
        print()

    def get_UCT_random_index(self, node):
        size = len(node.act_vect_)
        if size != len(node.weights_):
            print(size, len(node.weights_))
        random_branch = random.choices([i for i in range(size)], weights=node.weights_, k=1)
        return random_branch[0]

    def update_values(self, node, mc_return):
        total_return = mc_return
        if node.num_visits_ == 0:
            node.first_MC_ = total_return
        node.num_visits_ += 1
        node.weighted_num_visits_ += 1
        # back until root is reached, the parent of root is None
        last_weighted_num_visits_ = 1
        while node.parent_act_ is not None:
            parent_act = node.parent_act_
            parent_act.num_visits_ += 1
            parent_act.weighted_num_visits_ += last_weighted_num_visits_ * parent_act.visit_update_weight

            total_return *= self.gamma_
            total_return += self.modify_reward(node.reward_)

            # incremental method, re-calculate the average reward
            parent_act.avg_return_ += (total_return - parent_act.avg_return_) / parent_act.num_visits_
            node = parent_act.parent_state_
            if node.parent_act_ is None:
                root_sub_return = total_return
            node.num_visits_ += 1
            node.weighted_num_visits_ += last_weighted_num_visits_ * parent_act.visit_update_weight
            last_weighted_num_visits_ = last_weighted_num_visits_ * parent_act.visit_update_weight
        # return root_sub_return

    def MC_sampling_depth(self, node, depth):
        """
        playing out process, with a limited depth
        """
        mc_return = self.leaf_value_
        self.sim_.set_state(node.next_candidate_components_, node.current_candidate_components_, node.state_)
        discount = 1
        final_return = None
        for i in range(0, depth):
            if self.sim_.is_terminal():
                mc_return += self.end_episode_value_
                break
            actions = self.sim_.get_actions()
            act_ID = int(random.random() * len(actions))
            r = self.sim_.act(actions[act_ID])

            mc_return += discount * self.modify_reward(r)
            if not final_return or (final_return < mc_return):
                final_return = mc_return
            discount *= self.gamma_
        self.sim_.get_state()
        if not final_return:
            final_return = 0
        return final_return

    def MC_sampling_terminal(self, node):
        """
        playing out process, until reach a terminal state
        """
        mc_return = self.end_episode_value_
        self.sim_.set_state(node.next_candidate_components_, node.current_candidate_components_, node.state_)
        reward_list = []
        discount = 1
        final_return = None
        final_return = self.sim_.default_policy(mc_return, self.gamma_, discount, reward_list,
                                                self.component_default_policy_, self.path_default_policy_)

        return final_return

    def modify_reward(self, orig):
        return orig

    def print_root_values(self):
        size = len(self.root_.node_vect_)
        for i in range(0, size):
            val = self.root_.node_vect_[i].avg_return_
            num_visit = self.root_.node_vect_[i].avg_return_
            print("(", self.root_.act_vect_.print(), ",", val, ",", num_visit, ") ")
        print(self.root_.is_terminal_)

    # def update_root_node(self, act, new_state):
    def update_root_node(self, act, new_state, keep_uct_tree=False):
        """
        Update the root to be one of its children after taking an environment action.
        :param act: the action taken
        :param new_state: the observed next state (necessary for stochastic transition)
        """
        flag = 0
        for act_ptr in range(len(self.root_.act_vect_)):
            if act.equal(self.root_.act_vect_[act_ptr]):
                flag = 1
                break
        if flag == 0:
            print("can not find the exited action")
            exit(0)
        # act_ptr = self.root_.act_vect_.index(act)
        # for i in range(len(self.root_.node_vect_)):
        #     if i != act_ptr:
        #         self.root_.node_vect_[i] = None
        if not keep_uct_tree:
            for i in range(len(self.root_.node_vect_)):
                if i != act_ptr:
                    self.root_.node_vect_[i] = None
        action_node = self.root_.node_vect_[act_ptr]

        if action_node.state_vect_ is None:
            depth = 1
            current = self.root_
            self.sim_.set_state(current.next_candidate_components_, current.current_candidate_components_,
                                current.state_)
            r = self.sim_.act(action_node)
            next_node = current.node_vect_[act_ptr].add_state_node(self.sim_.get_state(),
                                                                   self.sim_.get_actions(),
                                                                   self.sim_.get_next_candidate_components(),
                                                                   self.sim_.get_current_candidate_components(),
                                                                   self.sim_.get_weights(),
                                                                   r, self.sim_.is_terminal())
            if -1 == self.max_depth_:
                mc_return = self.MC_sampling_terminal(next_node)
            else:
                mc_return = self.MC_sampling_depth(next_node, self.max_depth_ - depth)
            current = next_node
            self.update_values(current, mc_return)

            self.root_ = current
            # need to reuse
            # self.root_.parent_act_ = None
            action_node = None
            return
        else:
            for s_node in action_node.state_vect_:
                if s_node.state_.equal(new_state):
                    self.root_ = s_node
                    # need to reuse
                    # self.root_.parent_act_ = None
                    action_node = None
                    return
        return None

    def clear_tree(self):
        self.root_ = None
        pass

    def terminal_root(self):
        return self.root_.is_terminal_

    def prune(self, _action):
        next_root = None
        size = len(self.root_.node_vect_)
        for i in range(0, size):
            if _action.equal(self.root_.act_vect_[i]):
                assert len(self.root_.node_vect_[i].state_vect_) == 1
                next_root = self.root_.node_vect_[i].state_vect_[0]
                del self.root_.node_vect_[i]
            else:
                tmp = self.root_.node_vect_[i]
                self.prune_action(tmp)

        assert next_root is not None
        self.root_ = next_root
        self.root_.parent_act_ = None

    def prune_state(self, _state):
        size_node = len(_state.node_vect_)
        for i in range(0, size_node):
            tmp = _state.node_vect_[i]
            self.prune_action(tmp)

        _state.node_vect_ = []
        del _state

    def prune_action(self, _action):
        size_node = len(_action.state_vect_)
        for i in range(0, size_node):
            tmp = _action.state_vect_[i]
            self.prune_state(tmp)
        _action.state_vect_ = []
        del _action

    def test_root(self, _state, _reward, _is_terminal):
        return self.root_ is not None \
               and (self.root_.reward_ == _reward) \
               and (self.root_.is_terminal_ == _is_terminal) \
               and self.root_.state_.equal(_state)

    def test_deterministic_property(self):
        if self.test_deterministic_property_state(self.root_):
            print("Deterministic Property Test passed!")
        else:
            print("Error in Deterministic Property  Test!")
            sys.exit(0)

    def test_deterministic_property_state(self, _state):
        act_size = len(_state.node_vect_)
        # we test all the actions under a _state
        for i in range(0, act_size):
            if not self.test_tree_structure_Action(_state.node_vect_[i]):
                return False
        return True

    def test_deterministic_property_action(self, action):
        state_size = len(action.state_vect_)
        # under a deterministic property, a on s can only generate one specific s'
        if state_size != 1:
            print("Error in Deterministic Property Test!")
            return False
        # test every state under an action
        for i in range(0, state_size):
            if not self.test_tree_structure_state(action.state_vect_[i]):
                # print("action test: False")
                return False
        # print("action test:True")
        return True

    def test_tree_structure(self):
        if self.test_tree_structure_state(self.root_):
            print("Tree Structure Test passed!")
        else:
            print("Error in Tree Structure Test!")
            sys.exit(1)

    def test_tree_structure_state(self, _state):
        act_visit_counter = 0
        act_size = len(_state.node_vect_)
        for i in range(0, act_size):
            act_visit_counter += _state.node_vect_[i].num_visits_
        # find out that whether the total number of _state' visit is
        # equal to the number of the visit of their parent action
        if (act_visit_counter + 1 != _state.num_visits_) and (not _state.is_terminal_):
            print("n(s) = sum_{a} n(s,a) + 1 failed ! \n Diff: ",
                  act_visit_counter + 1 - _state.num_visits_,
                  "\nact: ", act_visit_counter + 1, "\nstate: ",
                  _state.num_visits_, "\nTerm: ",
                  _state.is_terminal_, "\nstate: ")
            _state.state_.print()
            print("")
            return False

        for i in range(0, act_size):
            if not self.test_tree_structure_Action(_state.node_vect_[i]):
                return False

        return True

    def test_tree_structure_Action(self, action):
        state_visit_counter = 0
        state_size = len(action.state_vect_)
        for i in range(0, state_size):
            state_visit_counter += action.state_vect_[i].num_visits_

        if state_visit_counter != action.num_visits_:
            print("n(s,a) = sum n(s') failed !")
            return False
        # avg
        # Q(s,a) = E {r(s') + gamma * sum pi(a') Q(s',a')}
        # Q(s,a) = sum_{s'} n(s') / n(s,a) * ( r(s')
        # + gamma * sum_{a'} (n (s',a') * Q(s',a') + first) / n(s'))
        value = 0
        for i in range(0, state_size):
            next_state = action.state_vect_[i]
            w = next_state.num_visits_ / float(action.num_visits_)
            next_value = next_state.first_MC_
            next_act_size = len(next_state.node_vect_)
            for j in range(0, next_act_size):
                next_value += next_state.node_vect_[j].num_visits_ * next_state.node_vect_[j].avg_return_
            next_value = next_value / next_state.num_visits_ * self.gamma_
            next_value += next_state.reward_
            value += w * next_value

        if (action.avg_return_ - value) * (action.avg_return_ - value) > 1e-10:
            print("value constraint failed !",
                  "avgReturn=", action.avg_return_, " value=", value)
            return False

        for i in range(0, state_size):
            if not self.test_tree_structure_state(action.state_vect_[i]):
                return False
        # print("test_tree_structure_Action pass")
        return True

    def print_state_visits(self, state_node):
        final_print = []
        act_size = len(state_node.node_vect_)
        if state_node.state_.graph == {} and len(state_node.state_.component_pool) > 3:
            print(state_node.state_.component_pool, "visit:", state_node.num_visits_)
        layer_perc = []
        for i in range(0, act_size):
            action_node = state_node.node_vect_[i]
            current_num_visit = self.print_state_visits(action_node.state_vect_[0])
            component_types = []
            if state_node.state_.graph == {} and 4 <= len(state_node.state_.component_pool) <= 7:
                for component in action_node.state_vect_[0].state_.component_pool:
                    if ('GND' not in component) and ('VIN' not in component) and ('VOUT' not in component):
                        component_types.append(get_component_type(component))
                sorted_component_types = sort_components(tuple(component_types), self.sim_.component_priority)
                layer_perc.append([sorted_component_types,
                                   current_num_visit * self.sim_.set_count_mapping[sorted_component_types]])
        if layer_perc:
            print(layer_perc)
            print([topo_count[1] for topo_count in layer_perc])
        return state_node.num_visits_

    def print_state_visits_level(self, state_node):
        Queue = [[[1, 2, 3, 4, 5], -1, state_node, 0]]
        final_counts = {}
        while Queue:
            _set = Queue.pop(0)
            parent_comps = _set[0]
            parent_depth = _set[1]
            state_node = _set[2]
            state_depth = _set[3]
            if len(state_node.state_.component_pool[3:]) == 5:
                # print(parent_comps[3:], parent_depth, '-->', state_node.state_.component_pool[3:], state_depth,
                #       ' visit ', state_node.num_visits_)
                print(state_node.state_.component_pool[3:], '\t', state_node.num_visits_)
                for component in state_node.state_.component_pool[3:]:
                    if ('GND' not in component) and ('VIN' not in component) and ('VOUT' not in component):
                        component_types.append(get_component_type(component))
                final_counts[tuple(
                    sort_components(tuple(component_types), self.sim_.component_priority))] = state_node.num_visits_
            act_size = len(state_node.node_vect_)
            for i in range(0, act_size):
                action_node = state_node.node_vect_[i]
                if action_node.state_vect_[0].state_.graph == {}:
                    Queue.append([state_node.state_.component_pool, state_depth,
                                  action_node.state_vect_[0], state_depth + 1])
            component_types = []
        for k, v in self.sim_.set_count_mapping.items():
            print(k, '\t', final_counts[k])

        return state_node.num_visits_


def sort_components(set_components, topo_priority):
    result = []
    reverse_topo_priority = {}
    for k, v in topo_priority.items():
        reverse_topo_priority[v] = k
    sorted_components_count = [0 for _ in range(len(topo_priority))]
    for i in set_components:
        sorted_components_count[topo_priority[i]] += 1
    # print(sorted_components_count)
    for i in range(len(topo_priority)):
        for counts in range(sorted_components_count[i]):
            result.append(reverse_topo_priority[i])
    return tuple(result)


def get_component_type(component):
    if component.startswith('L'):
        ret = 'L'
    elif component.startswith('C'):
        ret = 'C'
    elif component.startswith('Sa'):
        ret = 'Sa'
    elif component.startswith('Sb'):
        ret = 'Sb'
    else:
        ret = component
    return ret
