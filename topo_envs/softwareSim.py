from UCT_5_UCB_unblc_restruct_DP_v1.ucts.TopoPlanner import TopoGenSimulator
from topo_envs.surrogateRewardSim import SurrogateRewardSimFactory


class SoftwareSimulatorFactory(SurrogateRewardSimFactory):
    def get_sim_init(self):
        return TopoGenSimulator
