filename="dataset_5_05_fix_comp.json"

for i in 0
do
	python auto_train.py -data $filename -target eff -seed $i -no_cuda -save_model lstm_model/5_05_fix_eff_cpu_0.pt
	python auto_train.py -data $filename -target vout -seed $i -no_cuda -save_model lstm_model/5_05_fix_vout_cpu_0.pt
done
