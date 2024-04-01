#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./models/tldr/reward/baseline
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi

if [ -d $OUTPUT  ] ; then exit; fi

mkdir -p $OUTPUT

deepspeed --include localhost:4,5,6,7 main.py \
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
   &> $OUTPUT/training.log
