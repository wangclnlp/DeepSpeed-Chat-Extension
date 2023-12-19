#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""

import os
import random

from tqdm import  tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import default_data_collator
from transformers import AutoTokenizer
from transformers.models.llama import LlamaTokenizer
import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine
from options import parse_args
import torch.distributed as dist

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from rlhf_llama.deepspeed_chat.training.utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from rlhf_llama.deepspeed_chat.training.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer
from rlhf_llama.deepspeed_chat.training.utils.module.lora import convert_lora_to_linear_layer
from rlhf_llama.deepspeed_chat.training.utils import datetime_utils


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if args.add_sft_loss:
        prompt_train_dataset, _ = create_prompt_dataset(
            args.local_rank, args.data_path, args.data_split,
            args.data_output_path, train_phase, args.seed, tokenizer,
            args.max_prompt_seq_len, train_phase3_with_sft_data=args.add_sft_loss,
            input_and_output_max_len=(args.max_prompt_seq_len+args.max_answer_seq_len))
    else:
        prompt_train_dataset, _ = create_prompt_dataset(
            args.local_rank, args.data_path, args.data_split,
            args.data_output_path, train_phase, args.seed, tokenizer,
            args.max_prompt_seq_len, train_phase3_with_sft_data=args.add_sft_loss)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size,
                                     tokenizer.pad_token_id,
                                     tokenizer)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_train_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_train_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    args = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', init_method='env://')

    print("local rank:", args.local_rank)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # assert not args.offload, "zero-offload is not currently supported but coming soon!"

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # create common tokenizer based on actor model
    if "baichuan" in args.tokenizer_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                                    fast_tokenizer=False,
                                                    trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                                    fast_tokenizer=False)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.add_eos_token_id = False
    tokenizer.padding_side = 'left'    # see rlhf_llama/deepspeed_chat/training/utils/data/data_utils.py  line: 245

    if args.use_comet_model:
        reward_tokenizer = None
        args.reward_model_name_or_path = None
    elif (args.reward_tokenizer_name_or_path is not None) and ("baichuan" in args.reward_tokenizer_name_or_path.lower()):
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_tokenizer_name_or_path,
                                                        fast_tokenizer=False,
                                                        trust_remote_code=True)
        reward_tokenizer.pad_token_id = 0
        reward_tokenizer.bos_token_id = 1
        reward_tokenizer.eos_token_id = 2
        reward_tokenizer.add_bos_token = True
        reward_tokenizer.add_eos_token = True
        reward_tokenizer.padding_side = 'right'
    elif args.reward_tokenizer_name_or_path is not None:
        # create common tokenizer based on critic model
        reward_tokenizer = LlamaTokenizer.from_pretrained(args.reward_tokenizer_name_or_path,
                                                        fast_tokenizer=False)
        reward_tokenizer.pad_token_id = 0
        reward_tokenizer.bos_token_id = 1
        reward_tokenizer.eos_token_id = 2
        reward_tokenizer.add_bos_token = True
        reward_tokenizer.add_eos_token = True 
        reward_tokenizer.padding_side = 'right'
    else:
        reward_tokenizer = None 

    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        reward_model_name_or_path=args.reward_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args,
        reward_tokenizer=reward_tokenizer)

    # args.end_of_conversation_token = "<|endoftext|>"
    args.end_of_conversation_token = ""

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, \
                Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}", args.global_rank)

        for step, (batch_prompt, batch_unsupervised) in enumerate(tqdm(
                zip(prompt_train_dataloader, unsupervised_train_dataloader), total=len(prompt_train_dataloader))):
            
            global_step += 1

            batch_prompt = to_device(batch_prompt, device)
            # from rlhf_llama.deepspeed_chat.training.utils import pdb ; pdb.set_trace()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_train_batch_size])

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_enable()

            if (not args.remove_critic_model) and (args.critic_gradient_checkpointing):
                rlhf_engine.critic.gradient_checkpointing_enable()

            if args.add_sft_loss:
                sft_input_ids = batch_prompt["sft_input_ids"]
                sft_attention_mask = batch_prompt["sft_attention_mask"]
                sft_labels = batch_prompt["sft_labels"]
                sft_input = {}
                sft_input["sft_input_ids"] = sft_input_ids      # prompt+output
                sft_input["sft_attention_mask"] = sft_attention_mask
                sft_input["sft_label"] = sft_labels             # [0,0,0,0,1,1,1,1] prompt+output

                out = trainer.generate_experience(batch_prompt['prompt'],
                                                batch_prompt['prompt_att_mask'],
                                                sft_input=sft_input)
            else:
                out = trainer.generate_experience(batch_prompt['prompt'],
                                                batch_prompt['prompt_att_mask'])

            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0
                average_kl_distance = 0

                critic_only = (epoch==0) and args.critic_go_first_steps and (step<args.critic_go_first_steps)
                if critic_only:
                    ppo_epochs = 1
                else:
                    ppo_epochs = args.ppo_epochs
                for ppo_ep in range(ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(zip(exp_dataset, unsup_dataset)):
                        if critic_only:
                            actor_loss, critic_loss, kl_distance = trainer.train_rlhf(exp_data, update_critic_only=True)
                        else:
                            actor_loss, critic_loss, kl_distance = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()
                        average_kl_distance += kl_distance.mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                average_reward = get_all_reduce_mean(average_reward).item()
                print_rank_0(
                    f'epoch: {epoch}|step: {global_step}|act_loss: {actor_loss_sum/inner_iter}|cri_loss: {critic_loss_sum/inner_iter}'
                    f'|unsuper_loss: {unsup_loss_sum/inner_iter}|average reward score: {average_reward/inner_iter}|'
                    f'KL distance: {average_kl_distance/inner_iter}', args.global_rank)

                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()
                # rlhf_engine.actor.enable_input_require_grads()

            if (global_step+1) % args.save_steps==0:
                checkpoint_save_path = args.output_dir+f"/checkpoint-epoch{epoch}-step{global_step}"

                if args.output_dir is not None:
                    print_rank_0('saving model ...')
                    rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
                    if not args.remove_critic_model:
                        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)

                    if args.enable_ema:
                        rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                            rlhf_engine.actor_ema)

                    if torch.distributed.get_rank() == 0:
                        save_hf_format(rlhf_engine.actor,
                                    tokenizer,
                                    args,
                                    sub_folder='actor')
                        if not args.remove_critic_model and args.save_critic_model:
                            save_hf_format(rlhf_engine.critic,
                                        tokenizer,
                                        args,
                                        sub_folder='critic')
                            
                        if args.enable_ema:
                            save_hf_format(rlhf_engine.actor_ema,
                                        tokenizer,
                                        args,
                                        sub_folder='actor_ema')

                    if args.actor_zero_stage == 3:
                        save_zero_three_model(rlhf_engine.actor,
                                            global_rank=args.global_rank,
                                            save_dir=os.path.join(checkpoint_save_path, 'actor'),
                                            zero_stage=args.actor_zero_stage)
                        if args.enable_ema:
                            save_zero_three_model(rlhf_engine.actor_ema,
                                                global_rank=args.global_rank,
                                                save_dir=os.path.join(checkpoint_save_path, 'actor_ema'),
                                                zero_stage=args.actor_zero_stage)
                            
                    if args.actor_zero_stage in [1,2]:
                        # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                        model_to_save = rlhf_engine.actor.module if hasattr(rlhf_engine.actor,
                                                                    'module') else rlhf_engine.actor
                        lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                        save_dir=os.path.join(checkpoint_save_path, 'actor')
                        os.makedirs(save_dir, exist_ok=True)
                        WEIGHTS_NAME = "pytorch_model.bin"
                        output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
                        torch.save(lean_state_dict, output_model_file)
                            
                    if not args.remove_critic_model:
                        if args.critic_zero_stage == 3 and args.save_critic_model:
                            save_zero_three_model(rlhf_engine.critic,
                                                global_rank=args.global_rank,
                                                save_dir=os.path.join(checkpoint_save_path, 'critic'),
                                                zero_stage=args.critic_zero_stage)
                        
                        if args.critic_zero_stage in [1,2] and args.save_critic_model:
                            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                            model_to_save = rlhf_engine.critic.module if hasattr(rlhf_engine.critic,
                                                                        'module') else rlhf_engine.critic
                            lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                            save_dir=os.path.join(checkpoint_save_path, 'critic')
                            os.makedirs(save_dir, exist_ok=True)
                            WEIGHTS_NAME = "pytorch_model.bin"
                            output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
                            torch.save(lean_state_dict, output_model_file)

    if args.output_dir is not None:
        print_rank_0('saving final model ...')
        checkpoint_save_path = args.output_dir+f"/final"
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        if not args.remove_critic_model:
            rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            if not args.remove_critic_model and args.save_critic_model:
                save_hf_format(rlhf_engine.critic,
                            tokenizer,
                            args,
                            sub_folder='critic')
                
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                  checkpoint_save_path, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                      checkpoint_save_path, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        
        if args.actor_zero_stage in [1,2]:
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
            model_to_save = rlhf_engine.actor.module if hasattr(rlhf_engine.actor,
                                                        'module') else rlhf_engine.actor
            lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
            save_dir=os.path.join(checkpoint_save_path, 'actor')
            os.makedirs(save_dir, exist_ok=True)
            WEIGHTS_NAME = "pytorch_model.bin"
            output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
            torch.save(lean_state_dict, output_model_file) 
        
                
        if not args.remove_critic_model:
            if args.critic_zero_stage == 3 and args.save_critic_model:
                save_zero_three_model(rlhf_engine.critic,
                                    global_rank=args.global_rank,
                                    save_dir=os.path.join(
                                    checkpoint_save_path, 'critic'),
                                    zero_stage=args.critic_zero_stage)
            
            if args.critic_zero_stage in [1,2] and args.save_critic_model:
                # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                model_to_save = rlhf_engine.critic.module if hasattr(rlhf_engine.critic,
                                                            'module') else rlhf_engine.critic
                lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                save_dir=os.path.join(checkpoint_save_path, 'critic')
                os.makedirs(save_dir, exist_ok=True)
                WEIGHTS_NAME = "pytorch_model.bin"
                output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
                torch.save(lean_state_dict, output_model_file)

if __name__ == "__main__":
    main()
