#!/usr/bin/env bash

start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}


export ACTOR_MODEL_PATH=<actor_model_path>
export CRITIC_MODEL_PATH=<critic_model_path> # Can be actor model and remove later.

export COMET_PATH=<comet_model_path> # Path to the comet model, like path/to/your/comet/model/model.ckpt


export TOKENIZER_PATH=$ACTOR_MODEL_PATH

export ACTOR_ZERO_STAGE=2
export CRITIC_ZERO_STAGE=3
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

deepspeed ./rlhf_llama/deepspeed_chat/training/step3_rlhf_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_split 0,0,1 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --tokenizer_name_or_path $TOKENIZER_PATH \
   --reward_model_name_or_path $CRITIC_MODEL_PATH \
   --reward_tokenizer_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning $Num_Padding_at_Beginning \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --gradient_accumulation_steps 1 \
   --generation_batch_numbers 2 \
   --ppo_epochs 1 \
   --save_steps 100 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
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
   --ppo_mini_epochs 1 \
   --offload \
   --use_comet_model \
   --comet_model_path $COMET_PATH \
   --devices_comet_model 0 \
   --comet_model_batch_size 1 \
   --reward_type seq \
   --add_sft_loss \
   --remove_critic_model

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
