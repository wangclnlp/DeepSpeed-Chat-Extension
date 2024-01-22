#!/bin/bash

CUR_DIR=`pwd`
ROOT=${CUR_DIR}
export PYTHONPATH=${ROOT}:${PYTHONPATH}

MODEL_PATH=<your_model_path>
TEST_DATA=<your_input.txt>
MAX_NEW_TOKENS=512
OUTPUT_PATH=<path_to_save_model_output>

deepspeed ./rlhf_llama/deepspeed_chat/training/step1_supervised_finetuning/predict.py \
            --model_name_or_path $MODEL_PATH \
            --max_new_tokens $MAX_NEW_TOKENS \
            --test_data $TEST_DATA \
            --batch_size 4 \
            --output_file $OUTPUT_PATH
