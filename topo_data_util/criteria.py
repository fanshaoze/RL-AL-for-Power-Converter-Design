# based on one duty cycle
# duty = '0.6'
# high_reward = lambda eff, v_out: duty in eff.keys() and (v_out[duty] < 80 or v_out[duty] > 120) and eff[duty] >= .8
# low_reward = lambda eff, v_out: duty in eff.keys() and eff[duty] <= .2

# based on any duty cycle, constraints become looser
#high_reward = lambda effs, v_outs: any(v_out < 80 or v_out > 120 for v_out in v_outs) and any(eff >= .8 for eff in effs)
#low_reward = lambda effs, v_outs: all(eff <= .2 for eff in effs)

from GetReward import calculate_reward
# TODO: must specify the target vout here!!!
high_reward = lambda eff, vout: calculate_reward({'efficiency': eff, 'output_voltage': vout}, target_vout=50) >= 0.6
low_reward = lambda eff, vout: calculate_reward({'efficiency': eff, 'output_voltage': vout}, target_vout=50) < 0.6
