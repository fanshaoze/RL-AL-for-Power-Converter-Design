filename="dataset_5_05_fix_comp.json"
methods="lstm"

for i in {0..9}
do
	for method in $methods
	do
		python auto_train.py -data $filename -target eff -encoding $method -seed $i -save-model lstm_model/5_05_fix_$method\_eff_$i.pt -train-ratio 0.9 -dev-ratio 0.05 -patience 50
		python auto_train.py -data $filename -target vout -encoding $method -seed $i -save-model lstm_model/5_05_fix_$method\_vout_$i.pt -train-ratio 0.9 -dev-ratio 0.05 -patience 50
	done
done
