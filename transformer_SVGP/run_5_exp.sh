filename="dataset_5_cleaned_label.json"
methods="lstm"

for i in {0..2}
do
	for method in $methods
	do
		python auto_train.py -data $filename -target eff -encoding $method -seed $i -train_ratio 0.6 -save_model lstm_model/5_all_lstm_eff_$i.pt
		#python auto_train.py -data $filename -target eff -encoding $method -seed $i -train_ratio 0.6 -duty_encoding mlp

		python auto_train.py -data $filename -target vout -encoding $method -seed $i -train_ratio 0.6 -save_model lstm_model/5_all_lstm_vout_$i.pt
		#python auto_train.py -data $filename -target vout -encoding $method -seed $i -train_ratio 0.6 -duty_encoding mlp
	done
done
