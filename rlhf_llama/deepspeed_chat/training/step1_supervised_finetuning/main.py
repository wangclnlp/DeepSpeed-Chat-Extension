#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import sys
from tqdm import  tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    get_scheduler,
)
from transformers import LlamaTokenizer, AutoTokenizer
from transformers import DataCollatorForSeq2Seq

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from rlhf_llama.deepspeed_chat.training.utils.data.data_utils import create_prompt_dataset
from rlhf_llama.deepspeed_chat.training.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from rlhf_llama.deepspeed_chat.training.utils.ds_utils import get_train_ds_config
from rlhf_llama.deepspeed_chat.training.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from rlhf_llama.deepspeed_chat.training.utils.model.model_utils import create_hf_model
from rlhf_llama.deepspeed_chat.training.utils.ewc import EWC
from rlhf_llama.deepspeed_chat.training.utils.predict import Predict
from options import parse_args

def main():
    args = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
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
    ds_config['deepspeed_multinode_launcher'] = 'standard'
    ds_config['wall_clock_breakdown'] = False

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    print_rank_0("load tokenizer………………………………")
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
        tokenizer.padding_side = 'left'
    elif tokenizer_need_extra_token_id():
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
                                                   fast_tokenizer=False)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'left'
    else:
        tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                        fast_tokenizer=False)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.add_eos_token_id = True
        tokenizer.add_bos_token_id = True
        tokenizer.padding_side = "left"

    # Prepare the data
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

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
        if "base" in args.previous_rlhf_model:
            for bin_path in os.listdir(args.previous_rlhf_model):
                if "0000" in bin_path:
                    previous_parameters_dict_sub = torch.load(args.previous_rlhf_model+"/"+bin_path,  map_location='cpu')
                    previous_parameters_dict.update(previous_parameters_dict_sub)
        else:
            previous_parameters_dict = torch.load(args.previous_rlhf_model+"/pytorch_model.bin",  map_location='cpu')

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
        if os.path.exists(f"{args.previous_round_before_sft_model}/previous_fisher.bin"):
            previous_fisher = torch.load(f"{args.previous_round_before_sft_model}/previous_fisher.bin")
            ewc.fisher = previous_fisher
            # compute the fisher number
            for n in current_model_parameters_dict.keys():
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


    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    # print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    global_training_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            global_loss = 0
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            global_loss += loss.item()

            # add parameter contraint and compute EWC loss for actor loss
            if args.iterative_alignment and (not using_hat_freeze):
                ewc_loss = ewc.compute_ewc_loss(model)
                model.backward(ewc_loss)
                global_loss += ewc_loss.item()

            model.step()
            global_training_step += 1
            print_rank_0(f'step: {step} loss:{global_loss}', args.global_rank)

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
                    if args.zero_stage == 3:
                        # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                        save_zero_three_model(model,
                                            args.global_rank,
                                            checkpoint_save_path,
                                            zero_stage=args.zero_stage)
                    if args.zero_stage in [1,2]:
                        # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                        model_to_save = model.module if hasattr(model,
                                                                    'module') else model
                        lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                        os.makedirs(checkpoint_save_path, exist_ok=True)
                        WEIGHTS_NAME = "pytorch_model.bin"
                        output_model_file = os.path.join(checkpoint_save_path, WEIGHTS_NAME)
                        torch.save(lean_state_dict, output_model_file)

        # # Evaluate perplexity on the validation set.
        # print_rank_0(
        #     f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
        #     args.global_rank)
        # perplexity = evaluation(model, eval_dataloader)
        # print_rank_0(f"ppl: {perplexity}", args.global_rank)
        model.tput_timer.update_epoch_count()

        checkpoint_save_path = args.output_dir+f"/epoch-{epoch}"

        if args.output_dir is not None:
            print_rank_0('saving the model ...', args.global_rank)
            model = convert_lora_to_linear_layer(model)
    
            if args.global_rank == 0:
                save_hf_format(model, tokenizer, args)

            if args.zero_stage == 3:
                # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                save_zero_three_model(model,
                                    args.global_rank,
                                    checkpoint_save_path,
                                    zero_stage=args.zero_stage)
            if args.zero_stage in [1,2]:
                # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                model_to_save = model.module if hasattr(model,
                                                            'module') else model
                lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                os.makedirs(checkpoint_save_path, exist_ok=True)
                WEIGHTS_NAME = "pytorch_model.bin"
                output_model_file = os.path.join(checkpoint_save_path, WEIGHTS_NAME)
                torch.save(lean_state_dict, output_model_file)

if __name__ == "__main__":
    main()
