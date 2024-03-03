#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from transformers.models.llama import LlamaTokenizer
from transformers import AutoTokenizer
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

from rlhf_llama.deepspeed_chat.training.utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from rlhf_llama.deepspeed_chat.training.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from rlhf_llama.deepspeed_chat.training.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from rlhf_llama.deepspeed_chat.training.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from rlhf_llama.deepspeed_chat.training.utils.model.model_utils import create_hf_model
from rlhf_llama.deepspeed_chat.training.utils.ewc import EWC
from rlhf_llama.deepspeed_chat.training.utils.predict import Predict

from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
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
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
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
        help=
        "Initial learning rate (after the potential warmup period) to use.",
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
    parser.add_argument("--beta",
                        type=float,
                        default=1e-1,
                        help="Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.")
    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.0,
                        help="conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
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
    parser.add_argument("--save_steps",
                        type=int,
                        default=-1,
                        help="Save checkpoint in steps.")
    ## Iterative Alignment
    parser.add_argument('--iterative_alignment',
                        action='store_true',
                        help='Enable Iteractive Alignment.')
    parser.add_argument('--previous_sft_model',
                        type=str,
                        help='model path of the previous sft model; compute the fisher for EWC.')
    parser.add_argument('--previous_round_after_sft_model',
                        type=str,
                        help='model path of the previous round model after sft.')
    parser.add_argument('--lamda_factor',
                        type=float,
                        default=1.0,
                        help='rl_loss + ewc_loss * lamda_factor')
    parser.add_argument('--ewc_max_weight',
                        type=float,
                        default=100.0,
                        help='Set max weight in  EWC.')
    parser.add_argument('--ewc_mse_factor',
                        type=float,
                        default=1e10,
                        help='(sft_model-previous_sft_model).mean()*ewc_mse_factor')
    parser.add_argument('--apply_original_ewc',
                        type=bool,
                        default=False,
                        help='Apply original EWC.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument('--offload_reference_model',
                        action='store_true',
                        help='Enable ZeRO Offload techniques for reference model.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
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
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    # Predict validation dataset setting
    parser.add_argument("--predict_steps",
                        type=int,
                        default=-1,
                        help="Predict file after steps.")
    parser.add_argument("--predict_max_new_tokens",
                        type=int,
                        default=512,
                        help="Max new tokens while predicting.")
    parser.add_argument("--predict_batch_size",
                        type=int,
                        default=16,
                        help="Batch size while predicting.")
    parser.add_argument("--predict_file",
                        type=str,
                        default='',
                        help="Path of the file to be predicted.")
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
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
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
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
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
        tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                    fast_tokenizer=True)

    def init_actor_model(model_path):
        model = create_hf_model(AutoModelForCausalLM,
                                model_path,
                                tokenizer,
                                ds_config,
                                disable_dropout=args.dropout==None)
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)

        AdamOptimizer = FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  betas=(0.9, 0.95))
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_train_epochs,
        )
        ds_config["zero_optimization"]["stage"] = 2
        ds_config["zero_optimization"]["offload_param"]["device"] = "none"
        ds_config["zero_optimization"]["offload_optimizer"]["device"] = "none"
        model, _, _, _ = deepspeed.initialize(model=model,
                                        optimizer=optimizer,
                                        args=args,
                                        config=ds_config,
                                        lr_scheduler=lr_scheduler,
                                        dist_init_required=True)
        return model

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                                tokenizer,
                                ds_config,
                                dropout=args.dropout)


    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)

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

    def evaluation(model, ref_model, tokenizer, eval_dataloader):
        model.eval()
        losses = 0
        all_step = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = to_device(batch, device)
            batch_size = batch['input_ids'].shape[0] // 2
            chosen_input_ids = batch['input_ids'][:batch_size]
            rejected_input_ids = batch['input_ids'][batch_size:]
            label_mask = (batch['input_ids'] != tokenizer.pad_token_id).int()
            for i in range(batch_size):
                divergence_ind = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero().squeeze(-1)
                if len(divergence_ind)> 0:
                    divergence_ind = divergence_ind[0]
                else:
                    divergence_ind = 0
                label_mask[i][:divergence_ind] = 0
                label_mask[i+batch_size][:divergence_ind] = 0
            with torch.no_grad():
                outputs = model(**batch)
                ref_outputs = ref_model(**batch)

            logps = get_batch_logps(outputs.logits, batch['input_ids'], label_mask)
            ref_logps = get_batch_logps(ref_outputs.logits, batch['input_ids'], label_mask)

            chosen_logps = logps[:batch_size]
            rejected_logps = logps[batch_size:]
            ref_chosen_logps = ref_logps[:batch_size]
            ref_rejected_logps = ref_logps[batch_size:]

            logits = args.beta * ((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps))
            loss = (- torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) - \
                        torch.nn.functional.logsigmoid(-logits) * args.label_smoothing).mean(0)
            losses += loss.float()
            all_step = step
        losses = losses / (all_step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        chosen_rewards = args.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = args.beta * (rejected_logps - ref_rejected_logps).detach()
        return chosen_rewards.mean().item(), rejected_rewards.mean().item(), losses.item()

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

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
     

    if args.iterative_alignment:
        ewc = EWC(args.lamda_factor, args.ewc_max_weight)
        print_rank_0("extract parameters of the previous rlhf model..................................")
        previous_parameters_dict = {}
        if "base" in args.previous_sft_model:
            for bin_path in os.listdir(args.previous_sft_model):
                if "0000" in bin_path:
                    previous_parameters_dict_sub = torch.load(args.previous_sft_model+"/"+bin_path,  map_location='cpu')
                    previous_parameters_dict.update(previous_parameters_dict_sub)
        else:
            previous_parameters_dict = torch.load(args.previous_sft_model+"/pytorch_model.bin",  map_location='cpu')

        print_rank_0("extract parameters of the current model..................................")
        current_model_parameters_dict = {}
        if "base" in args.model_name_or_path:
            for bin_path in os.listdir(args.model_name_or_path):
                if "0000" in bin_path:
                    current_model_parameters_dict_sub = torch.load(args.model_name_or_path+"/"+bin_path,  map_location='cpu')
                    current_model_parameters_dict.update(current_model_parameters_dict_sub)
        else:
            current_model_parameters_dict = torch.load(args.model_name_or_path+"/pytorch_model.bin",  map_location='cpu')

        # check if the fisher from the previous round exists.
        print_rank_0("compute the fisher..................................")
        if os.path.exists(f"{args.previous_round_after_sft_model}/previous_fisher.bin"):
            previous_fisher = torch.load(f"{args.previous_round_after_sft_model}/previous_fisher.bin")
            ewc.fisher = previous_fisher
            # compute the fisher number
            if len(current_model_parameters_dict.keys()) < len(previous_parameters_dict.keys()):
                for n in current_model_parameters_dict.keys():
                    fp = current_model_parameters_dict[n.replace("module.", "")]
                    previous_fp = previous_parameters_dict[n.replace("module.", "")]
                    # compute the different between rlhf model and previous rlhf model.
                    ewc.fisher[n.replace("module.", "")] += (((fp - previous_fp)*args.ewc_mse_factor) ** 2).mean().item()
            else:
                for n in previous_parameters_dict.keys():
                    fp = current_model_parameters_dict[n.replace("module.", "")]
                    previous_fp = previous_parameters_dict[n.replace("module.", "")]
                    # compute the different between rlhf model and previous rlhf model.
                    ewc.fisher[n.replace("module.", "")] += (((fp - previous_fp)*args.ewc_mse_factor) ** 2).mean().item()
        else:
            # compute the fisher number
            for n in current_model_parameters_dict.keys():
                fp = current_model_parameters_dict[n.replace("module.", "")]
                previous_fp = previous_parameters_dict[n.replace("module.", "")]
                # compute the different between sft model and previous sft model.
                # from rlhf_llama.deepspeed_chat.training.utils import pdb ; pdb.set_trace()
                ewc.fisher[n.replace("module.", "")] = (((fp - previous_fp)*args.ewc_mse_factor) ** 2).mean().item()

        # <--baseline method(HAT-Freeze): freezing parameters--begin>
        using_hat_freeze = False
        if using_hat_freeze:
            freeze_rate = 0.2
            sorted_fisher = dict(sorted(ewc.fisher.items(), key=lambda item: item[1], reverse=True))
            select_freeze_parameter = list(sorted_fisher.keys())[:int(len(sorted_fisher) * freeze_rate)] 

            for n, p in model.named_parameters():
                if n.replace("module.", "") in select_freeze_parameter:
                    p.requires_grad = False
        # <--baseline method(HAT-Freeze): freezing parameters--end>

        # save the fisher in this round
        torch.save(ewc.fisher, f"{args.model_name_or_path}/previous_fisher.bin")
        # using softmax to normalize the fisher
        norm_values = torch.softmax(torch.Tensor(list(ewc.fisher.values())), dim=-1)


        # <--baseline method(HAT-Random): random weights--begin>
        using_hat_random = False
        if using_hat_random:
            random_fisher_value = torch.rand_like(norm_values)
            norm_values = torch.softmax(random_fisher_value, dim=-1)
        # <--baseline method(HAT-Random): random weights--end>

        ewc.fisher = dict(zip(ewc.fisher.keys(), norm_values.tolist()))

        print_rank_0("end...delete parameters dict..................................")
        del previous_parameters_dict
        ewc.reference_model_parameters = current_model_parameters_dict
        # del current_model_parameters_dict
        import gc
        gc.collect()


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
 
    ref_model, *_ = deepspeed.initialize(model=ref_model,
                                         config=ref_ds_config)
    ref_model.eval()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating rewards, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # chosen_rewards, rejected_rewards, eval_loss = evaluation(model, ref_model, tokenizer, eval_dataloader)
    # print_rank_0(f"chosen: {chosen_rewards}, rejected: {rejected_rewards}, loss: {eval_loss}", args.global_rank)

    global_training_step = 0
    for epoch in range(args.num_train_epochs):
        # Calculate fiser matrix
        if args.iterative_alignment and args.apply_original_ewc:
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
                batch_size = batch['input_ids'].shape[0] // 2
                batch = {k:v[:batch_size] for k,v in batch.items()}
                batch = to_device(batch, device)
                outputs = model(**batch, use_cache=False)
                label_mask = (batch['input_ids'] != tokenizer.pad_token_id).int()
                logps = get_batch_logps(outputs.logits, batch['input_ids'], label_mask)
                logps = logps.mean()
                model.backward(logps, retain_graph=True)
                ewc.update_fisher_matrix_with_grad(model)
            model.eval()

        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
            batch = to_device(batch, device)
            batch_size = batch['input_ids'].shape[0] // 2
            chosen_input_ids = batch['input_ids'][:batch_size]
            rejected_input_ids = batch['input_ids'][batch_size:]
            label_mask = (batch['input_ids'] != tokenizer.pad_token_id).int()
            for i in range(batch_size):
                divergence_ind = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero().squeeze(-1)
                if len(divergence_ind)> 0:
                    divergence_ind = divergence_ind[0]
                else:
                    divergence_ind = 0
                label_mask[i][:divergence_ind] = 0
                label_mask[i+batch_size][:divergence_ind] = 0
            outputs = model(**batch, use_cache=False)
            with torch.no_grad():
                ref_outputs = ref_model(**batch)

            logps = get_batch_logps(outputs.logits, batch['input_ids'], label_mask)
            ref_logps = get_batch_logps(ref_outputs.logits, batch['input_ids'], label_mask)

            chosen_logps = logps[:batch_size]
            rejected_logps = logps[batch_size:]
            ref_chosen_logps = ref_logps[:batch_size]
            ref_rejected_logps = ref_logps[batch_size:]

            logits = args.beta * ((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps))
            loss = (- torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) - \
                        torch.nn.functional.logsigmoid(-logits) * args.label_smoothing).mean(0)
            model.backward(loss)

            # add parameter contraint and compute EWC loss for actor loss
            if args.iterative_alignment and (not using_hat_freeze):
                ewc_loss = ewc.compute_ewc_loss(model, apply_original_ewc=args.apply_original_ewc)
                if ewc_loss != 0:
                    model.backward(ewc_loss)
                    loss += ewc_loss.item()

            if args.print_loss:
                if step%20 == 0:
                    if args.iterative_alignment and (not using_hat_freeze):
                        print_rank_0(
                            f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss: {loss}, ewc_los: {ewc_loss.item() if type(ewc_loss)==torch.tensor else ewc_loss}"
                        )
                    else:
                        print_rank_0(
                            f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss: {loss}"
                        )

            model.step()
            global_training_step += 1

            with torch.no_grad():
                if args.predict_steps != -1 and global_training_step % args.predict_steps == 0:
                    old_tokenizer_padding_side = tokenizer.padding_side
                    tokenizer.padding_side = 'left'
                    predict_output = os.path.join(args.output_dir, 'predicts', f'epoch_{epoch}_step_{global_training_step}.txt')
                    if args.local_rank in [-1, 0]:
                        if not os.path.exists(os.path.dirname(predict_output)):
                            os.makedirs(os.path.dirname(predict_output))
                    print_rank_0("Predicting...")
                    assert os.path.exists(args.predict_file)
                    Predict(args.predict_max_new_tokens, args.predict_file, 0.75, 0.95, \
                            args.predict_batch_size, predict_output, local_rank=args.local_rank).predict(model, device, tokenizer)
                    tokenizer.padding_side = old_tokenizer_padding_side

            if global_training_step % args.save_steps == 0:
                checkpoint_save_path = args.output_dir+f"/epoch-{epoch}-step{global_training_step}"
                if args.output_dir is not None:
                    print_rank_0('saving the model ...', args.global_rank)
                    model = convert_lora_to_linear_layer(model)
                    if args.global_rank == 0:
                        save_hf_format(model, tokenizer, args)
                    if args.zero_stage in [0, 1, 2]:
                        # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                        model_to_save = model.module if hasattr(model,
                                                                    'module') else model
                        lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                        os.makedirs(checkpoint_save_path, exist_ok=True)
                        WEIGHTS_NAME = "pytorch_model.bin"
                        output_model_file = os.path.join(checkpoint_save_path, WEIGHTS_NAME)
                        torch.save(lean_state_dict, output_model_file)

                    if args.zero_stage == 3:
                        # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                        save_zero_three_model(model,
                                            args.global_rank,
                                            checkpoint_save_path,
                                            zero_stage=args.zero_stage)


        checkpoint_save_path = args.output_dir+f"/epoch-{epoch}"

        if args.output_dir is not None:
            print_rank_0('saving the model ...', args.global_rank)
            model = convert_lora_to_linear_layer(model)

            if args.global_rank == 0:
                save_hf_format(model, tokenizer, args)

            if args.zero_stage in [0, 1, 2]:
                # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                model_to_save = model.module if hasattr(model,
                                                            'module') else model
                lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                os.makedirs(checkpoint_save_path, exist_ok=True)
                WEIGHTS_NAME = "pytorch_model.bin"
                output_model_file = os.path.join(checkpoint_save_path, WEIGHTS_NAME)
                torch.save(lean_state_dict, output_model_file)

            if args.zero_stage == 3:
                # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                save_zero_three_model(model,
                                      args.global_rank,
                                      checkpoint_save_path,
                                      zero_stage=args.zero_stage)

        # Evaluate perplexity on the validation set.
        # print_rank_0(
        #     f"***** Evaluating rewards, Epoch {epoch+1}/{args.num_train_epochs} *****",
        #     args.global_rank)
        # chosen_rewards, rejected_rewards, eval_loss = evaluation(model, ref_model, tokenizer, eval_dataloader)
        # print_rank_0(f"chosen: {chosen_rewards}, rejected: {rejected_rewards}, loss: {eval_loss}", args.global_rank)
        model.tput_timer.update_epoch_count()



if __name__ == "__main__":
    main()
