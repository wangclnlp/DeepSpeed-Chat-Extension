#!/usr/bin/env bash

start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

MODEL_PATH=<your_model_path>
TOKENIZER_PATH=$MODEL_PATH

ZERO_STAGE=<zero_stage>
OUTPUT=<path_to_save_model>
DATA_PATH=<reward_data>

DATA_CACHE=./data_cache

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

mkdir -p $OUTPUT

deepspeed ./rlhf_llama/deepspeed_chat/training/step2_dpo_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_split 0,1,0 \
   --model_name_or_path $MODEL_PATH \
   --data_output_path $DATA_CACHE \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-6 \
   --weight_decay 0. \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --offload_reference_model \
   --output_dir $OUTPUT \
   --print_loss \
   --beta 0.1 \
   &> $OUTPUT/training.log

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}


