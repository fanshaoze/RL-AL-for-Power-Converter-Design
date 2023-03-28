filename="dataset_5_05_cleaned_label.json"
extra=""

for i in {0..4}
do
	python auto_train.py -data $filename -target valid -encoding lstm -seed $i -save_model lstm_model/5_lstm_valid_$i.pt $extra
done
