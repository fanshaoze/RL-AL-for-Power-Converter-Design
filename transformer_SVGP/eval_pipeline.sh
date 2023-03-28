for i in {0..4}
do
	python test_model.py -pretrained_model lstm_model/5_all_lstm_eff_$i.pt -pretrained_validity_model lstm_model/5_lstm_valid_$i.pt -data_test dataset_5_05_cleaned_label_test_$i.json -target eff
done
