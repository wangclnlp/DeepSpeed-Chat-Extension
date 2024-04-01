#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./models/tldr/reward/cos_sim/beta1_30_beta3_-25
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi

if [ -d $OUTPUT  ] ; then exit; fi

mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3 --master_port 29501 main.py \
   --data_path data/summarize-from-feedback/dataset/comparisons/ \
   --data_output_path ./data_cache \
   --data_split 0,1,0 \
   --model_name_or_path models/tldr/sft/epoch-3 \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 1 \
   --max_seq_len 1024 \
   --learning_rate 1e-6 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --dtype bf16 \
   --output_dir $OUTPUT \
   --save_epochs 1 \
   --method_for_controlling_scale_of_reward cosine_similarity \
   --beta1_for_controlling_scale_of_reward 30 \
   --beta2_for_controlling_scale_of_reward 0.001 \
   --beta3_for_controlling_scale_of_reward -25 \
   &> $OUTPUT/training.log
