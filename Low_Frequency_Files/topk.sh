for eff in 9
do
	for vout in 9
	do
		python topoExp.py --model transformer --eff-model transformer_SVGP/save_model/5_comp_eff_$eff.pt --vout-model transformer_SVGP/save_model/5_comp_vout_$vout.pt --traj 30 50 100 200 --k-list 10 30 50 100 --seed-range 0 50 --output topk_$eff\_$vout
	done
done
