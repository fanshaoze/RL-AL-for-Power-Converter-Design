#filename="dataset_5_05_strict_valid_set.json"
filename="dataset_5_valid_set.json"
#filename="dataset_5_05_cleaned_label.json"
methods="lstm"

extra="-n_layers 5 -n_heads 16 -mlp_layers 128 64 64 16"

for i in {0..9}
do
	for method in $methods
	do
		python auto_train.py -data $filename -target eff -encoding $method -seed $i -save-model lstm_model/5_all_$method\_test_eff_$i.pt $extra
	done
done
