#filename="dataset_5_05_strict_valid_set.json"
#filename="dataset_5_valid_set.json"
filename="dataset_5_cleaned_label.json"
methods="lstm"
extra=""

for i in {0..4}
do
	for method in $methods
	do
		python auto_train.py -data $filename -target vout -encoding $method -seed $i -train_ratio 0.2 $extra
		python auto_train.py -data $filename -target vout -encoding $method -seed $i -train_ratio 0.4 $extra
	done
done
