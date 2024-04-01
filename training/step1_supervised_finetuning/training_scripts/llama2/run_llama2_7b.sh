#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=models/alpaca/sft/
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3,4,5,6,7 main.py \
   --data_path data/alpaca_farm/alpaca_instructions \
   --data_output_path ./data_cache \
   --data_split 1,0,0 \
   --model_name_or_path models/meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 16 \
   --gradient_accumulation_steps 1 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --num_train_epochs 3 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --print_loss \
   --dtype bf16 \
   --save_epochs 1\
   &> $OUTPUT/training.log

