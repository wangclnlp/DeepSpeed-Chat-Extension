#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import sys
import socket
import time

import numpy as np
from tqdm import  tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    get_scheduler,
)
from transformers.models.llama import LlamaTokenizer
from transformers import AutoTokenizer
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from rlhf_llama.deepspeed_chat.training.utils.model.model_utils import create_critic_model
from rlhf_llama.deepspeed_chat.training.utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from rlhf_llama.deepspeed_chat.training.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from rlhf_llama.deepspeed_chat.training.utils.ds_utils import get_train_ds_config
from rlhf_llama.deepspeed_chat.training.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from rlhf_llama.deepspeed_chat.training.utils import datetime_utils
from options import parse_args

def print_args(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape)).item()
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

def main():
    args = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    # dist.init_process_group(backend='nccl', init_method='env://')
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    # print("global rank is:", args.global_rank)
    # print("world size is:", torch.distributed.get_world_size())

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['deepspeed_multinode_launcher'] = 'standard' 
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['wall_clock_breakdown'] = False


    # args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # tokenizer.pad_token = tokenizer.eos_token
    print("load tokenizer………………………………")
    def tokenizer_need_extra_token_id():
        keyword_list = ['llama', 'ziya', 'gpt2', 'rm']
        for keyword in keyword_list:
            if keyword in args.model_name_or_path.lower():
                return True
        return False
    
    if "baichuan" in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                    fast_tokenizer=False,
                                                    trust_remote_code=True)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'right'
        
    elif tokenizer_need_extra_token_id():
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
                                                   fast_tokenizer=False)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'right'
    else:
        tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=False)

    print("load dataset………………………………")
    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len, args=args)


    print("create dataloader………………………………")
    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    print("load model………………………………")
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout,
                                   dropout=args.head_dropout,
                                   is_reward=True)

    print_args(rm_model) 
    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
    print_args(rm_model)


    def evaluation_reward(model, eval_dataloader, gpt_annotated_score=False):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = None
        rejected_scores = 0
        
        if gpt_annotated_score:
            mse_loss = 0
            loss_counter = 0

        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            if scores is None:
                scores = outputs["chosen_mean_scores"]
                rejected_scores = outputs["rejected_mean_scores"]
            else:
                scores = torch.concat([scores, outputs["chosen_mean_scores"]])
                rejected_scores = torch.concat([rejected_scores, outputs["rejected_mean_scores"]])
            
            if gpt_annotated_score:
                mse_loss += outputs["loss"]*len(batch["gpt_score"])
                loss_counter += len(batch["gpt_score"])

        total_predictions = scores.shape[0]
        scores_ave = scores.sum().float() / total_predictions
        rejected_scores_ave = rejected_scores.sum().float() / total_predictions
        acc = correct_predictions / total_predictions
        score_std = np.std(scores.cpu().detach().half().numpy())
        try:
            acc = get_all_reduce_mean(acc).item()
            scores_ave = get_all_reduce_mean(scores_ave).item()
            score_std = get_all_reduce_mean(score_std).item()
            rejected_scores_ave = get_all_reduce_mean(rejected_scores_ave).item()
        except:
            pass
        
        if gpt_annotated_score:
            return scores_ave, rejected_scores_ave, acc, score_std, mse_loss/loss_counter
        else:
            return scores_ave, rejected_scores_ave, acc, score_std

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    print("start deepspeed initizlize………………")
    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        mean_loss = 0
        rm_model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            global_step += 1
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]

            correct_predictions = (chosen > rejected).sum() * 1.0 / chosen.shape[0]
            reward = outputs["chosen_mean_scores"].mean().float()
            r_reward = outputs["rejected_mean_scores"].mean().float()
            print_rank_0(f'step: {step} loss:{loss}, '
                         f'correct_predictions: {correct_predictions}, '
                         f'reward: {reward} '
                         f'r_reward: {r_reward} ',
                         args.global_rank)
            
            if (step+1)%args.save_steps==0:
                checkpoint_save_path = args.output_dir+f"/checkpoint-{step}"

                if args.output_dir is not None:
                    print_rank_0('saving model ...', args.global_rank)
                    
                    rm_model = convert_lora_to_linear_layer(rm_model)

                    if args.global_rank == 0:
                        save_hf_format(rm_model, tokenizer, args)
                    if args.zero_stage == 3:
                        # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
                        save_zero_three_model(rm_model,
                                            args.global_rank,
                                            checkpoint_save_path,
                                            zero_stage=args.zero_stage)
                    if args.zero_stage in [1,2]:
                        # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                        model_to_save = rm_model.module if hasattr(rm_model,
                                                                    'module') else rm_model
                        lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                        os.makedirs(checkpoint_save_path, exist_ok=True)
                        WEIGHTS_NAME = "pytorch_model.bin"
                        output_model_file = os.path.join(checkpoint_save_path, WEIGHTS_NAME)
                        torch.save(lean_state_dict, output_model_file)

            if (global_step + 1) % args.eval_steps == 0:
                if args.gpt_annotated_score:
                    reward_score, rejected_scores, acc, score_std, mse_loss = evaluation_reward(rm_model, eval_dataloader, args.gpt_annotated_score)
                    print_rank_0(f"Eval/epoch: {epoch+1}, Eval/step: {global_step+1} Eval/reward_score: {reward_score}, Eval/score_std: {score_std}, \
                                Eval/rejected_scores: {rejected_scores}, Eval/acc: {acc}, Mse: {mse_loss}")
                else:
                    reward_score, rejected_scores, acc, score_std = evaluation_reward(rm_model, eval_dataloader, args.gpt_annotated_score)
                    print_rank_0(f"Eval/epoch: {epoch+1}, Eval/step: {global_step+1} Eval/reward_score: {reward_score}, Eval/score_std: {score_std}, \
                                Eval/rejected_scores: {rejected_scores}, Eval/acc: {acc}")
                rm_model.train()
                # chosen_last_scores (higher is better) : -0.37704116106033325, reject_last_scores (higher is better) 
                # : -0.41206246614456177,  acc (higher is better) : 0.564919114112854

        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)

        
        rm_model.tput_timer.update_epoch_count()

        if args.output_dir is not None:
            print_rank_0('saving model ...', args.global_rank)
            checkpoint_save_path = args.output_dir+f"/epoch-{epoch}-checkpoint-final"
            
            rm_model = convert_lora_to_linear_layer(rm_model)

            if args.global_rank == 0:
                save_hf_format(rm_model, tokenizer, args)
            if args.zero_stage == 3:
                # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
                save_zero_three_model(rm_model,
                                        args.global_rank,
                                        checkpoint_save_path,
                                        zero_stage=args.zero_stage)
            if args.zero_stage in [1,2]:
                # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                model_to_save = rm_model.module if hasattr(rm_model,
                                                            'module') else rm_model
                lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                os.makedirs(checkpoint_save_path, exist_ok=True)
                WEIGHTS_NAME = "pytorch_model.bin"
                output_model_file = os.path.join(checkpoint_save_path, WEIGHTS_NAME)
                torch.save(lean_state_dict, output_model_file)


if __name__ == "__main__":
    main()
