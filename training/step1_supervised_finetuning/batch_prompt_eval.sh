#!/bin/bash
root_dir=$*

for f in `find -L $root_dir -path '*pytorch_model.bin'`
do
  f_dir=`dirname $f`
  echo $f_dir
  python ../step1_supervised_finetuning/prompt_eval_batch.py --model_name_or_path $f_dir --dataset alpaca --test_json data/alpaca_farm/alpaca_farm_evaluation.json --batch_size 8 --output_dir $f_dir/alpaca_farm_evaluation --max_seq_len 512
done
