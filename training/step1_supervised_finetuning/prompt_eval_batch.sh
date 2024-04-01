# dirs=( models/alpaca/ppo/cos_sim/beta1_20_beta3_-15/step-200/actor/ ./models/alpaca/ppo/baseline/step-200/actor ./models/alpaca/ppo/len_rat/beta1_10_beta3_-5/step-200/actor models/alpaca/sft/epoch-3/ )
# 
# 
# for d in ${dirs[@]}
# do
# 	# d=$d/epoch-1/
# 	echo $d
# 	for t in 0 0.25 0.5 0.75 1; do
# 		for c in 0 1 2 ;do
# 			output_dir=$d/predictions_temperature/temp${t}_count${c}
# 			if [ -d $output_dir ]; then continue; fi
# 			python ../step1_supervised_finetuning/prompt_eval_batch.py --model_name_or_path $d --dataset alpaca --test_json data/alpaca_farm/alpaca_farm_evaluation.json --dtype bf16 --batch_size 32 --max_seq_len 512 --temperature $t --output_dir $output_dir
# 		done
# 	done
# done



for d in models/alpaca/ppo/baseline_analyze/step-*/actor
do
	# d=$d/epoch-1/
	echo $d
	output_dir=$d/predictions_temp0/
	if [ -d $output_dir ]; then continue; fi
	python ../step1_supervised_finetuning/prompt_eval_batch.py --model_name_or_path $d --dataset alpaca --test_json data/alpaca_farm/alpaca_farm_evaluation.json --dtype bf16 --batch_size 32 --max_seq_len 512 --temperature 0 --output_dir $output_dir
done
