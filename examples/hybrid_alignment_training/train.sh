#!/usr/bin/env bash

###################################################
INI_SFT_MODEL_PATH=/path/to/the/pretrained/model
REWARD_MODEL_PATH=/path/to/the/reward/model
OUTPUT_ROOT=/path/to/the/root/of/checkpoints
LOG_ROOT=/path/to/the/root/of/log
SFT_DATA_ROOT=/path/to/the/sft/data/splits
RLHF_DATA_ROOT=/path/to/the/sft/data/splits
EWC_MAX_WEIGHT=$1
data_id=1
EPOCH_NUM=3
SPLIT_NUM=10
###################################################

start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export LANG="zh_CN.UTF-8"
CUR_DIR=`pwd`
ROOT=${CUR_DIR}
export PYTHONPATH=${ROOT}:${PYTHONPATH}

run_init_sft () 
{
    echo "================================"
    start_time=`date +%Y%m%d%H%M%S`
    echo "start ${start_time}"
    echo "Running SFT......"
    echo DATA_PATH: $DATA_PATH
    echo SFT_MODEL_PATH: $SFT_MODEL_PATH
    echo OUTPUT: $OUTPUT
    echo LOG_PATH: $LOG_PATH
    echo EPOCH_NUM: $EPOCH_NUM
    deepspeed --master_port 12346 --include=localhost:0,1,2,3,4,5,6,7 ./rlhf_llama/deepspeed_chat/training/step1_supervised_finetuning/main.py \
        --data_path $DATA_PATH \
        --data_split 1,0,0 \
        --model_name_or_path $SFT_MODEL_PATH \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --per_device_eval_batch_size 4 \
        --max_seq_len 1024 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --num_train_epochs $EPOCH_NUM  \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 1234 \
        --gradient_checkpointing \
        --zero_stage 3 \
        --deepspeed \
        --output_dir $OUTPUT \
        --dtype bf16 \
        > $LOG_PATH 2>&1
    end_time=`date +%Y%m%d%H%M%S`
    echo ${end_time}
    echo "Done."
    echo "================================"
}

run_sft () 
{
    echo "================================"
    start_time=`date +%Y%m%d%H%M%S`
    echo "start ${start_time}"
    echo "Running SFT......"
    echo DATA_PATH: $DATA_PATH
    echo SFT_MODEL_PATH: $SFT_MODEL_PATH
    echo PREVIOUS_RLHF_MODEL: $PREVIOUS_RLHF_MODEL
    echo PREVIOUS_ROUND_BEFORE_SFT: $PREVIOUS_ROUND_BEFORE_SFT
    echo OUTPUT: $OUTPUT
    echo LOG_PATH: $LOG_PATH
    echo EPOCH_NUM: $EPOCH_NUM
    deepspeed --master_port 12346 --include=localhost:0,1,2,3,4,5,6,7 ./rlhf_llama/deepspeed_chat/training/step1_supervised_finetuning/main.py \
        --data_path $DATA_PATH \
        --data_split 1,0,0 \
        --model_name_or_path $SFT_MODEL_PATH \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 4 \
        --max_seq_len 1024 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --num_train_epochs $EPOCH_NUM  \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 1234 \
        --gradient_checkpointing \
        --zero_stage 2 \
        --deepspeed \
        --output_dir $OUTPUT \
        --dtype bf16 \
	--offload \
        --iterative_alignment \
        --previous_rlhf_model $PREVIOUS_RLHF_MODEL \
        --previous_round_before_sft_model $PREVIOUS_ROUND_BEFORE_SFT \
        > $LOG_PATH 2>&1
    end_time=`date +%Y%m%d%H%M%S`
    echo ${end_time}
    echo "Done."
    echo "================================"
}

run_rlhf () 
{
    echo "================================"
    start_time=`date +%Y%m%d%H%M%S`
    echo "start ${start_time}"
    echo "Running RLHF......"
    echo DATA_PATH: $DATA_PATH
    echo ACTOR_MODEL_PATH: $ACTOR_MODEL_PATH
    echo REWARD_MODEL_PATH: $REWARD_MODEL_PATH
    echo CRITIC_MODEL_PATH: $CRITIC_MODEL_PATH
    echo PREVIOUS_SFT_MODEL: $PREVIOUS_SFT_MODEL
    echo PREVIOUS_ROUND_AFTER_SFT: $PREVIOUS_ROUND_AFTER_SFT
    echo OUTPUT: $OUTPUT
    echo LOG_PATH: $LOG_PATH
    deepspeed --master_port 12346 --include=localhost:0,1,2,3,4,5,6,7 ./rlhf_llama/deepspeed_chat/training/step3_rlhf_finetuning/main.py \
        --data_path $DATA_PATH \
        --data_split 0,0,1 \
        --actor_model_name_or_path $ACTOR_MODEL_PATH \
        --reward_model_name_or_path $REWARD_MODEL_PATH \
        --critic_model_name_or_path $CRITIC_MODEL_PATH \
        --tokenizer_name_or_path $ACTOR_MODEL_PATH \
        --reward_tokenizer_name_or_path $REWARD_MODEL_PATH \
        --data_output_path ./data_cache \
        --num_padding_at_beginning 0 \
        --per_device_train_batch_size 8 \
        --per_device_mini_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --generation_batch_numbers 8 \
        --ppo_epochs 1 \
        --save_steps 500 \
        --max_answer_seq_len 512 \
        --max_prompt_seq_len 256 \
        --actor_learning_rate 1e-5 \
        --actor_weight_decay 0.1 \
        --critic_weight_decay 0.1 \
        --num_train_epochs 1 \
        --lr_scheduler_type cosine \
        --disable_actor_dropout \
        --num_warmup_steps 100 \
        --deepspeed --seed 1234 \
        --actor_gradient_checkpointing \
        --critic_gradient_checkpointing \
        --actor_zero_stage 2 \
        --critic_zero_stage 3 \
        --output_dir $OUTPUT \
        --dtype bf16 \
        --ppo_mini_epochs 1 \
        --remove_critic_model \
        --reward_type seq \
        --offload \
        --iterative_alignment \
        --ewc_max_weight $EWC_MAX_WEIGHT \
        --ewc_mse_factor 1e10 \
        --previous_sft_model $PREVIOUS_SFT_MODEL \
        --previous_round_after_sft_model $PREVIOUS_ROUND_AFTER_SFT \
        > $LOG_PATH 2>&1
    end_time=`date +%Y%m%d%H%M%S`
    echo ${end_time}
    echo "Done."
    echo "================================"
}

for i in `seq 1 10`; do
    DATA_PATH=$SFT_DATA_ROOT/sft_${SPLIT_NUM}_$data_id
    if [ "$i" -eq 1 ] ; then
        SFT_MODEL_PATH=$INI_SFT_MODEL_PATH
    else
        SFT_MODEL_PATH=$OUTPUT_ROOT/rl_$((i-1))/final/actor
    fi

    OUTPUT=$OUTPUT_ROOT/sft_$i
    LOG_PATH=$LOG_ROOT/sft_$i.log
    PREVIOUS_RLHF_MODEL=$OUTPUT_ROOT/sft_$((i-1))/epoch-$((EPOCH_NUM-1))/
    PREVIOUS_ROUND_BEFORE_SFT=$OUTPUT_ROOT/rl_$((i-2))/final/actor
    if [ "$i" -eq 1 ] ; then
        if [ ! -d $OUTPUT ] ; then
            run_init_sft
        else
        echo "$OUTPUT exists. Skipped......"
        fi
    else
        if [ ! -d $OUTPUT ] ; then
            run_sft
        else
        echo "$OUTPUT exists. Skipped......"
        fi
    fi
    echo 'copy: ' $OUTPUT/config.json '->' $OUTPUT/epoch-$((EPOCH_NUM-1))/
    cp $OUTPUT/config.json $OUTPUT/epoch-$((EPOCH_NUM-1))/
    echo 'copy: ' $OUTPUT/tokenizer.model '->' $OUTPUT/epoch-$((EPOCH_NUM-1))/
    cp $OUTPUT/tokenizer.model $OUTPUT/epoch-$((EPOCH_NUM-1))/
    

    DATA_PATH=$RLHF_DATA_ROOT/rlhf_${SPLIT_NUM}_$data_id
    ACTOR_MODEL_PATH=$OUTPUT_ROOT/sft_$i/epoch-$((EPOCH_NUM-1))/
    #REWARD_MODEL_PATH=$REWARD_MODEL_PATH
    CRITIC_MODEL_PATH=$INI_SFT_MODEL_PATH
    PREVIOUS_SFT_MODEL=$SFT_MODEL_PATH
    PREVIOUS_ROUND_AFTER_SFT=$OUTPUT_ROOT/sft_$((i-1))/epoch-$((EPOCH_NUM-1))/

    rm -rf $OUTPUT_ROOT/sft_$((i-1))/epoch-$((EPOCH_NUM-2))/
    rm -rf $OUTPUT_ROOT/sft_$((i-1))/epoch-$((EPOCH_NUM-3))/

    OUTPUT=$OUTPUT_ROOT/rl_$i
    LOG_PATH=$LOG_ROOT/rl_$i.log
    if [ ! -d $OUTPUT ] ; then
        run_rlhf
    else
	echo "$OUTPUT exists. Skipped......"
    fi
    echo 'copy: ' $OUTPUT/actor/config.json '->' $OUTPUT/final/actor
    cp $OUTPUT/actor/config.json $OUTPUT/final/actor
    echo 'copy: ' $OUTPUT/actor/tokenizer.model '->' $OUTPUT/final/actor
    cp $OUTPUT/actor/tokenizer.model $OUTPUT/final/actor
    data_id=$((data_id+1))
done

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
