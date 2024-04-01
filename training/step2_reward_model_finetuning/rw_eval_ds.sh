for d in models/alpaca/reward/len_rat/beta1_10_beta3_-5/epoch-1/ models/alpaca/reward/baseline/epoch-1/ models/alpaca/reward/cos_sim/beta1_20_beta3_-15/epoch-1/
do
	deepspeed rw_eval_ds.py --model_name_or_path $d --dataset alpaca --test_json data/alpaca_farm/human_comparisons/train_analyze.json --output_dir $d/train_analyze --dtype bf16 --batch_size 128 --max_seq_len 512
done
