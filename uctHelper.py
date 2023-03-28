# import logging
#
# from UCT_5_UCB_unblc_restruct_DP_v1.ucts.uct import UCTPlanner, StateNode
#
#
# """
# These utility functions are supposed to work for all UCT classes.
# """
# def postorder_traverse(state: StateNode, fn, depth):
#     """
#     Postorder traversal of the tree rooted at state
#     Apply fn once visited
#     """
#     for act in state.node_vect_:
#         for next_state in act.state_vect_:
#             postorder_traverse(next_state, fn, depth + 1)
#     fn(state, depth)
#
# def preorder_traverse(state: StateNode, fn, depth):
#     """
#     Postorder traversal of the tree rooted at state
#     Apply fn once visited
#     """
#     fn(state, depth)
#     for act in state.node_vect_:
#         for next_state in act.state_vect_:
#             preorder_traverse(next_state, fn, depth + 1)
#
# def traverse_tree(uct_tree: UCTPlanner, fn, order='post'):
#     """
#     Traverse the UCT tree in postorder (traverse the children of a node before the node itself)
#     :param uct_tree:
#     :param fn:
#     :return:
#     """
#     if order == 'post':
#         postorder_traverse(uct_tree.root_, fn, depth=0)
#     elif order == 'pre':
#         preorder_traverse(uct_tree.root_, fn, depth=0)
#     else:
#         raise Exception('unknown traverse order ' + order)
#
# def recompute_tree_rewards(uct_tree: UCTPlanner, sim):
#     def update_reward(state: StateNode, depth):
#         parent_act = state.parent_act_
#         sim.set_state(state=state)
#         if parent_act is None:
#             # at the root, nothing to do
#             return
#         # elif len(state.node_vect_) == 1 and state.node_vect_[0].type == 'Terminal':
#         elif sim.is_terminal():
#             # terminal node, need to recompute reward
#             print('find a terminal and update')
#             state.reward_ = sim.get_reward(state.state_)
#             parent_act.avg_return_ = state.reward_
#         else:
#             # internal nods, pass the average return up
#             sum_ret = 0
#             for act in state.node_vect_:
#                 sum_ret += act.avg_return_ * act.num_visits_
#
#             # fixme hacking for deterministic transitions
#             parent_act.avg_return_ = 1. * sum_ret / state.num_visits_
#
#     traverse_tree(uct_tree, update_reward, order='post')
#
# def print_avg_return_in_tree(uct_tree: UCTPlanner):
#     def printer(state: StateNode, depth):
#         # print the average return of the *parent* of this state
#         # (this is easier to implement than printing all its children nodes)
#         parent_act = state.parent_act_
#         if parent_act is not None:
#             logging.info("    " * depth + str(parent_act.avg_return_))
#
#     traverse_tree(uct_tree, printer, order='pre')


import logging

from ucts.uct import StateNode, UCTPlanner

"""
These utility functions are supposed to work for all UCT classes.
"""


def postorder_traverse(state: StateNode, fn, depth):
    """
    Postorder traversal of the tree rooted at state
    Apply fn once visited
    """
    for act in state.node_vect_:
        for next_state in act.state_vect_:
            postorder_traverse(next_state, fn, depth + 1)
    fn(state, depth)


def preorder_traverse(state: StateNode, fn, depth):
    """
    Preorder traversal of the tree rooted at state
    Apply fn once visited
    """
    fn(state, depth)
    for act in state.node_vect_:
        for next_state in act.state_vect_:
            preorder_traverse(next_state, fn, depth + 1)


def traverse_tree(uct_tree: UCTPlanner, fn, order='post'):
    """
    Traverse the UCT tree in postorder (traverse the children of a node before the node itself)
    :param uct_tree:
    :param fn:
    :return:
    """
    if order == 'post':
        postorder_traverse(uct_tree.root_, fn, depth=0)
    elif order == 'pre':
        preorder_traverse(uct_tree.root_, fn, depth=0)
    else:
        raise Exception('unknown traverse order ' + order)


def recompute_tree_rewards(uct_tree: UCTPlanner, sim):
    def update_reward(state: StateNode, depth):
        parent_act = state.parent_act_

        if parent_act is None:
            # at the root, nothing to do
            return
        sim.set_state(_current_candidate_components=state.current_candidate_components_,
                      _next_candidate_components=state.next_candidate_components_, state=state.state_)
        if sim.is_terminal():
            # terminal node, need to recompute reward
            state.reward_ = sim.get_reward(state.state_)
            parent_act.avg_return_ = state.reward_
        else:
            # internal nods, pass the average return up

            # if not state.node_vect_:
            if state.node_vect_:
                # state.reward_ = sim.get_reward(state.state_)
                # parent_act.avg_return_ = state.reward_
                # else:
                sum_ret = 0
                for act in state.node_vect_:
                    sum_ret += act.avg_return_ * act.num_visits_

                # fixme hacking for deterministic transitions
                parent_act.avg_return_ = 1. * sum_ret / state.num_visits_

    traverse_tree(uct_tree, update_reward, order='post')


def print_avg_return_in_tree(uct_tree: UCTPlanner):
    def printer(state: StateNode, depth):
        # print the average return of the *parent* of this state
        # (this is easier to implement than printing all its children nodes)
        parent_act = state.parent_act_
        if parent_act is not None:
            logging.info("    " * depth + str(parent_act.avg_return_))

    traverse_tree(uct_tree, printer, order='pre')
