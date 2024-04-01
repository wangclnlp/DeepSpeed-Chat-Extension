#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
    AutoTokenizer,
    AutoModel
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

from dschat.utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.perf import print_throughput


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    # Reference: https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-1,
        help="Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    # LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use."
    )
    # low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    # Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step2_tensorboard")
    # Checkpoints saving
    parser.add_argument("--save_epochs",
                        type=int,
                        default=999999,
                        help="Epochs to save checkpoints.")
    parser.add_argument("--save_last",
                        action='store_true',
                        help="Save the final model.")
    # Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    # Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')

    # Method for controlling scale of the reward
    parser.add_argument(
        "--method_for_controlling_scale_of_reward",
        type=str,
        default=None,
        choices=['length_ratio', 'cosine_similarity'],
        help='Select the method for controlling the scale of the reward.')
    # parser.add_argument(
    #     "--model_path_for_cos_sim_controlling_scale_of_reward",
    #     type=str,
    #     default=None,
    #     help='Model path for cosine similarity controlling scale of the reward.')
    parser.add_argument(
        "--beta1_for_controlling_scale_of_reward",
        type=float,
        default=6,
        help="Control delta_star=beta1/(similarity+beta2)+beta3."
    )
    parser.add_argument(
        "--beta2_for_controlling_scale_of_reward",
        type=float,
        default=2,
        help="Control delta_star=beta1/(similarity+beta2)+beta3."
    )
    parser.add_argument(
        "--beta3_for_controlling_scale_of_reward",
        type=float,
        default=0,
        help="Control delta_star=beta1/(similarity+beta2)+beta3."
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


# Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
def get_batch_logps(logits, input_ids, label_mask):
    labels = input_ids.clone() * label_mask
    assert logits.shape[:-1] == labels.shape, \
        "Logits (batch and sequence length dim) and labels must have the same shape."
    labels = labels[:, 1:]
    label_mask = label_mask[:, 1:]
    logits = logits[:, :-1, :]
    per_token_logps = torch.gather(logits.log_softmax(-1),
                                   dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * label_mask).sum(-1)


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout)

    # DS Config for ref model
    ref_zero_stage = args.zero_stage
    if ref_zero_stage != 3:
        # If it is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
        ref_zero_stage = 0
    ref_ds_config = get_eval_ds_config(args.offload_reference_model,
                                       args.dtype, ref_zero_stage)
    ref_ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ref_ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps
    ref_ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=args.dtype,
                                            stage=ref_zero_stage)
    ref_ds_eval_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ref_ds_eval_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps
    ref_model = create_hf_model(AutoModelForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ref_ds_eval_config,
                                dropout=args.dropout)
    # End of DS config for ref model

    if args.method_for_controlling_scale_of_reward == 'cosine_similarity':
        ref_sim_model_name_or_path = 'models/google-bert/bert-base-cased'
        ref_sim_tokenizer = AutoTokenizer.from_pretrained(
            ref_sim_model_name_or_path)
        ref_sim_tokenizer.truncation_side = 'left'
        # DS Config for ref_sim model
        # ref_sim_zero_stage = args.zero_stage
        # if ref_sim_zero_stage != 3:
        #     # If it is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref_sim model
        #     ref_sim_zero_stage = 0
        ref_sim_zero_stage = 0
        ref_sim_ds_config = get_eval_ds_config(offload=False,
                                               dtype=args.dtype,
                                               stage=ref_sim_zero_stage)
        ref_sim_ds_config[
            'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
        ref_sim_ds_config[
            'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
        ref_sim_ds_eval_config = get_eval_ds_config(offload=False,
                                                    dtype=args.dtype,
                                                    stage=ref_sim_zero_stage)
        ref_sim_ds_eval_config[
            'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
        ref_sim_ds_eval_config[
            'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
        ref_sim_model = AutoModel.from_pretrained(ref_sim_model_name_or_path)
        # End of DS config for ref_sim model
        ref_sim_model, *_ = deepspeed.initialize(
                model=ref_sim_model, config=ref_sim_ds_config)
        ref_sim_model.eval()
        ref_sim_train_phase = 2
        ref_sim_train_dataset, ref_sim_eval_dataset = create_prompt_dataset(
            args.local_rank, args.data_path, args.data_split,
            args.data_output_path, ref_sim_train_phase, args.seed, ref_sim_tokenizer,
            512, end_of_conversation_token="",)

        # DataLoaders creation:
        ref_sim_data_collator = DataCollatorReward()
        if args.local_rank == -1:
            ref_sim_train_sampler = RandomSampler(ref_sim_train_dataset)
        else:
            ref_sim_train_sampler = DistributedSampler(ref_sim_train_dataset)
        ref_sim_train_dataloader = DataLoader(ref_sim_train_dataset,
                                              collate_fn=ref_sim_data_collator,
                                              sampler=ref_sim_train_sampler,
                                              batch_size=args.per_device_train_batch_size)

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

    # Prepare the data
    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len, end_of_conversation_token=tokenizer.eos_token,)

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
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation(model, ref_model, tokenizer, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            batch_size = batch['input_ids'].shape[0] // 2
            chosen_input_ids = batch['input_ids'][:batch_size]
            rejected_input_ids = batch['input_ids'][batch_size:]
            label_mask = (batch['input_ids'] != tokenizer.pad_token_id).int()
            for i in range(batch_size):
                divergence_ind = (chosen_input_ids[i] !=
                                  rejected_input_ids[i]).nonzero().squeeze(-1)
                if len(divergence_ind) > 0:
                    divergence_ind = divergence_ind[0]
                else:
                    divergence_ind = 0
                label_mask[i][:divergence_ind] = 0
                label_mask[i + batch_size][:divergence_ind] = 0
            with torch.no_grad():
                outputs = model(**batch)
                ref_outputs = ref_model(**batch)

            logps = get_batch_logps(outputs.logits, batch['input_ids'],
                                    label_mask)
            ref_logps = get_batch_logps(ref_outputs.logits, batch['input_ids'],
                                        label_mask)

            chosen_logps = logps[:batch_size]
            rejected_logps = logps[batch_size:]
            ref_chosen_logps = ref_logps[:batch_size]
            ref_rejected_logps = ref_logps[batch_size:]

            logits = args.beta * ((chosen_logps - ref_chosen_logps) -
                                  (rejected_logps - ref_rejected_logps))
            loss = (- torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) -
                    torch.nn.functional.logsigmoid(-logits) * args.label_smoothing).mean(0)
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        chosen_rewards = args.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = args.beta * (rejected_logps -
                                        ref_rejected_logps).detach()
        return chosen_rewards.mean().item(), rejected_rewards.mean().item(
        ), losses.item()

    def save_model(model, tokenizer, args, sub_folder=""):
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, sub_folder=sub_folder)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  os.path.join(args.output_dir, sub_folder),
                                  zero_stage=args.zero_stage)

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

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

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    ref_model, *_ = deepspeed.initialize(model=ref_model, config=ref_ds_config)
    ref_model.eval()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    torch.distributed.barrier()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating rewards, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # chosen_rewards, rejected_rewards, eval_loss = evaluation(
    #     model, ref_model, tokenizer, eval_dataloader)
    # print_rank_0(
    #     f"chosen: {chosen_rewards}, rejected: {rejected_rewards}, loss: {eval_loss}",
    #     args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()

        if args.method_for_controlling_scale_of_reward == 'cosine_similarity':
            assert len(ref_sim_train_dataloader) == len(train_dataloader)
            ref_sim_train_dataloader_iter = iter(ref_sim_train_dataloader)

        import time
        print("Total steps: {}".format(len(train_dataloader)))
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)

            if args.method_for_controlling_scale_of_reward == 'cosine_similarity':
                ref_sim_batch = to_device(
                    next(ref_sim_train_dataloader_iter), device)
                with torch.no_grad():
                    ref_sim_output = ref_sim_model(**ref_sim_batch)
                seq_cls = ref_sim_output[0][:, 0, :]
                seq_cls_nor = torch.nn.functional.normalize(seq_cls, p=2, dim=1)

            batch_size = batch['input_ids'].shape[0] // 2
            chosen_input_ids = batch['input_ids'][:batch_size]
            rejected_input_ids = batch['input_ids'][batch_size:]
            label_mask = (batch['input_ids'] != tokenizer.pad_token_id).int()
            similarity_batch = []
            for i in range(batch_size):
                divergence_ind = (chosen_input_ids[i] !=
                                  rejected_input_ids[i]).nonzero().squeeze(-1)
                if len(divergence_ind) > 0:
                    divergence_ind = divergence_ind[0]
                else:
                    divergence_ind = 0
                label_mask[i][:divergence_ind] = 0
                label_mask[i + batch_size][:divergence_ind] = 0

                if args.method_for_controlling_scale_of_reward == 'length_ratio':
                    len0 = label_mask[i].sum()
                    len1 = label_mask[i+batch_size].sum()
                    longer_len = max(len0, len1)
                    shorter_len = min(len0, len1)
                    if longer_len != 0:
                        similarity = shorter_len / longer_len
                    else:
                        similarity = 1
                    similarity_batch.append(similarity)
                elif args.method_for_controlling_scale_of_reward == 'cosine_similarity':
                    similarity = torch.nn.functional.cosine_similarity(
                        seq_cls_nor[i].unsqueeze(0), seq_cls_nor[i+batch_size].unsqueeze(0)).item()
                    similarity_batch.append(similarity)
            similarity_batch = torch.Tensor(
                similarity_batch).to(device)

            outputs = model(**batch, use_cache=False)
            with torch.no_grad():
                ref_outputs = ref_model(**batch)

            logps = get_batch_logps(outputs.logits, batch['input_ids'],
                                    label_mask)
            ref_logps = get_batch_logps(ref_outputs.logits, batch['input_ids'],
                                        label_mask)

            chosen_logps = logps[:batch_size]
            rejected_logps = logps[batch_size:]
            ref_chosen_logps = ref_logps[:batch_size]
            ref_rejected_logps = ref_logps[batch_size:]

            logits = args.beta * ((chosen_logps - ref_chosen_logps) -
                                  (rejected_logps - ref_rejected_logps))

            if args.method_for_controlling_scale_of_reward != None:
                with torch.no_grad():
                    delta_star = args.beta1_for_controlling_scale_of_reward / \
                        (similarity_batch+args.beta2_for_controlling_scale_of_reward) + args.beta3_for_controlling_scale_of_reward
                loss_ori = (- torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) -
                            torch.nn.functional.logsigmoid(-logits) * args.label_smoothing).mean(0)
                loss_con = (- torch.nn.functional.logsigmoid(delta_star-logits) * (1 - args.label_smoothing) -
                            torch.nn.functional.logsigmoid(logits-delta_star) * args.label_smoothing).mean(0)
                loss = (loss_ori + loss_con)
            else:
                loss = (- torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) -
                        torch.nn.functional.logsigmoid(-logits) * args.label_smoothing).mean(0)
            if args.print_loss and args.local_rank in [-1,0]:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, \
                      Loss = {loss}, Similarity = {similarity_batch}"
                )
            model.backward(loss)
            model.step()
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start,
                                 args.global_rank)
                hours, rem = divmod((end-start)*(len(train_dataloader)-step-1), 3600)
                minutes, seconds = divmod(rem, 60)
                print("Time needed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating rewards, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        chosen_rewards, rejected_rewards, eval_loss = evaluation(
            model, ref_model, tokenizer, eval_dataloader)
        print_rank_0(
            f"chosen: {chosen_rewards}, rejected: {rejected_rewards}, loss: {eval_loss}",
            args.global_rank)
        model.tput_timer.update_epoch_count()

        if (epoch+1)%args.save_epochs==0 and args.output_dir is not None:
            save_model(model, tokenizer, args, sub_folder="epoch-{}".format(epoch+1))

    if args.save_last and args.output_dir is not None:
        save_model(model, tokenizer, args)


if __name__ == "__main__":
    main()
