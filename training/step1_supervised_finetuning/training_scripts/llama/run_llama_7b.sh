#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama_7b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
cp $0 $OUTPUT

deepspeed --include localhost:$CUDA_CARDS main.py \
   --data_path data/tldr_3_filtered \
   --data_output_path ./data_cache \
   --data_split 1,0,0 \
   --model_name_or_path meta-llama/Llama-7b-hf \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 1024 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --output_dir $OUTPUT \
   --print_loss \
   &> $OUTPUT/training.log
