#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step2_llama_7b_epoch3_lr3e-6
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:$CUDA_CARDS --master_port 29501 main.py \
   --data_path ./data/alpaca_farm/human_comparisons \
   --data_output_path ./data_cache \
   --data_split 0,1,0 \
   --model_name_or_path ../step1_supervised_finetuning/output_step1_llama_7b/ \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 3e-6 \
   --weight_decay 0.0 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --offload \
   --output_dir $OUTPUT \
   --method_for_controlling_scale_of_reward cosine_similarity \
   &> $OUTPUT/training.log

# --model_name_or_path meta-llama/Llama-1b-hf \
