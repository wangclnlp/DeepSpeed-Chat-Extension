#!/usr/bin/env bash

start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

MODEL_PATH=<sft_model_for_training_reward_model>
TOKENIZER_PATH=$MODEL_PATH

ZERO_STAGE=3
OUTPUT=<path_to_save_model>
DATA_PATH=<data_for_reward>

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

mkdir -p $OUTPUT

export Num_Padding_at_Beginning=0 # this is model related

deepspeed ./rlhf_llama/deepspeed_chat/training/step2_reward_model_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_split 0,1,0 \
   --model_name_or_path $MODEL_PATH \
   --num_padding_at_beginning $Num_Padding_at_Beginning \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --disable_dropout \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --save_steps 500 \
   --eval_steps 100 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
