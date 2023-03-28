for eff in 9
do
	for vout in 9
	do
		python topoQueryExp.py --model transformer --eff-model-seed $eff --vout-model-seed $vout --seed-range 0 50 --output al_$eff\_$vout\_tmp
	done
done
