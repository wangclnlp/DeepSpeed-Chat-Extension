#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=./models/alpaca/sft/epoch-3
CRITIC_MODEL_PATH=./models/alpaca/reward/baseline/epoch-1/
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./models/alpaca/ppo/baseline
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
if [ -d $OUTPUT ] ; then exit ; fi
mkdir -p $OUTPUT

Actor_Lr=1e-5
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path data/alpaca_farm/alpaca_rlhf \
   --data_split 0,0,1 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 384 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 4 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --actor_dropout 0.0 \
   --num_warmup_steps 100 \
   --deepspeed --seed 42 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --output_dir $OUTPUT \
   --dtype bf16 \
   --save_steps 20 \
   --num_train_steps 200 \
   --add_eot_token \
   --print_answers \
   --print_answers_interval 10 \
    &> $OUTPUT/training.log

# --offload_reference_model \

