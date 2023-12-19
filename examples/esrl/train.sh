#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

export ACTOR_MODEL_PATH=<your_policy_model>
export REWARD_MODEL_PATH=<your_reward_model>
export CRITIC_MODEL_PATH=$REWARD_MODEL_PATH

export TOKENIZER_PATH=$ACTOR_MODEL_PATH    # actor and critic tokenizer
export REWARD_TOKENIZER_PATH=$REWARD_MODEL_PATH  # reward tokenizer

export ACTOR_ZERO_STAGE=<actor_zero_stage>
export CRITIC_ZERO_STAGE=<critic_zero_stage>
export OUTPUT=<path_to_save_model>
export DATA_PATH=<data_for_rlhf>

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

export Num_Padding_at_Beginning=0 # this is model related

export Actor_Lr=1e-5
export Critic_Lr=5e-6

deepspeed --master_port 12346 --include=localhost:0,1,2,3,4,5,6,7 ./rlhf_llama/deepspeed_chat/training/step3_rlhf_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_split 0,0,1 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --reward_model_name_or_path $REWARD_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --tokenizer_name_or_path $TOKENIZER_PATH \
   --reward_tokenizer_name_or_path $REWARD_TOKENIZER_PATH \
   --data_output_path ./data_cache \
   --num_padding_at_beginning $Num_Padding_at_Beginning \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --generation_batch_numbers 2 \
   --ppo_epochs 1 \
   --save_steps 100 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 768 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --output_dir $OUTPUT \
   --dtype bf16 \
   --ppo_mini_epochs 4 \
   --reward_type lex \
   --offload \
   --dynamic_sampling \


end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
