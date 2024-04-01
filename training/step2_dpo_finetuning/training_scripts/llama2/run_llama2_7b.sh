#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./models/alpaca/dpo/baseline
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi

echo $OUTPUT
if [ -d $OUTPUT ] ; then continue; fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path data/alpaca_farm/human_comparisons/ \
   --data_split 0,1,0 \
   --model_name_or_path models/alpaca/sft/epoch-3 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-6 \
   --weight_decay 0.0 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --dtype bf16 \
   --print_loss \
   --beta 0.1 \
   --save_epochs 1 \
   &> $OUTPUT/training.log
