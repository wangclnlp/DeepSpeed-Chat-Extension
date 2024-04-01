#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import torch

import deepspeed
from dschat.utils.model.model_utils import create_critic_model
from dschat.utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator
from dschat.utils.ds_utils import get_eval_ds_config

import json
import os
from torch import multiprocessing
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['alpaca', 'tldr'],
        default='alpaca',
    )
    parser.add_argument(
        "--test_json",
        type=str,
        help="Path to test dataset in json format containing keys 'chosen' and 'rejected'.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Dir of the output of test dataset.",
        required=False,
        default=None
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    # parser.add_argument(
    #     "--add_eot_token",
    #     action='store_true',
    #     help="Add <|endoftext|> as additional special token to tokenizer")
    # parser.add_argument(
    #     "--cards",
    #     type=int,
    #     default=-1,
    #     nargs='+',
    #     help="Cards to eval.")
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Data type')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size.")
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=512,
                        help="Max sequence length.")


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def load_stuff(model_name_or_path, num_padding_at_beginning,
               additional_special_tokens):

    tokenizer = load_hf_tokenizer(model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    ref_zero_stage = 3
    ref_ds_config = get_eval_ds_config(offload=False,
                                       dtype=args.dtype,
                                       stage=ref_zero_stage)
    ref_ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=args.dtype,
                                            stage=ref_zero_stage)
    model = create_critic_model(model_name_or_path,
                                tokenizer,
                                ref_ds_eval_config,
                                num_padding_at_beginning,
                                dropout=0.,
                                rlhf_training=True,
                                zero_stage=ref_zero_stage)
    model, *_ = deepspeed.initialize(model=model, config=ref_ds_config)
    model.eval()

    return model, tokenizer


def process_single_data(d, dataset='alpaca', end_of_conversation_token=""):
    if dataset == 'alpaca':
        if not d['instruction'].endswith(d['input']):
            prompt = "\n\nHuman: " + \
                d['instruction'] + d['input'] + "\n\nAssistant: "
        else:
            prompt = "\n\nHuman: " + d['instruction'] + "\n\nAssistant: "
        chosen = prompt + \
            (d['output_1'] if d['preference'] == 1 else d['output_2']) + end_of_conversation_token
        rejected = prompt + \
            (d['output_1'] if d['preference'] == 2 else d['output_2']) + end_of_conversation_token
        return chosen, rejected
    if dataset == 'tldr':
        prompt = "\n\nHuman: Your task is to generate a short summary of a post.\n\nPost: "\
              f"{d['prompt']}\n\nSummary: \n\nAssistant: "
        chosen = prompt + d['chosen'] + end_of_conversation_token
        rejected = prompt + d['rejected'] + end_of_conversation_token
        return chosen, rejected


def tokenize_chosen_rejected(data, tokenizer, args):
    chosen, rejected = [], []
    count = 0
    print_num = 1
    if args.local_rank in [-1, 0]:
        data = tqdm(data)
    for d in data:
        c, r =  process_single_data(d, dataset=args.dataset, end_of_conversation_token=tokenizer.eos_token)
        chosen.append(c)
        rejected.append(r)
        count += 1
        if count % args.batch_size == 0:
            seq = chosen + rejected
            batch = tokenizer(seq,
                              max_length=args.max_seq_len,
                              padding="max_length",
                              truncation=True,
                              return_tensors="pt")
            if print_num > 0:
                print(seq[0])
                print(batch['input_ids'][0])
                print_num -= 1
            chosen, rejected = [], []
            count = 0
            yield batch, seq
    if count != 0:
        seq = chosen + rejected
        # print(seq)
        batch = tokenizer(seq,
                            max_length=args.max_seq_len,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt")
        yield batch, seq

def run(data, device, args):
    args.add_eot_token = False
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning,
                                     additional_special_tokens)
    rm_model.to(device)

    scores = []
    # Run inference
    data_gen = tokenize_chosen_rejected(data, tokenizer, args)
    for batch, seq in data_gen:
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = rm_model(**batch)
        bs = len(seq)//2
        for i in range(bs):
            scores.append({
                'chosen': seq[i],
                'rejected': seq[i+bs],
                'c_score': outputs['chosen_mean_scores'][i].item(),
                'r_score': outputs['rejected_mean_scores'][i].item()
            })
    return scores


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.test_json)
    if args.output_dir == None:
        args.output_dir = os.path.join(args.model_name_or_path, 'predictions')
    # assert not os.path.exists(os.path.join(args.output_dir, 'predictions'))
    assert not os.path.exists(args.output_dir)

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    with open(args.test_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = run(data, device, args)
    if args.local_rank in [0, -1]:
        results_score_only = []
        for r in results:
            results_score_only.append([r['c_score'], r['r_score']])
        accuracy = [rs[0] > rs[1]
                       for rs in results_score_only].count(True) / len(results_score_only)
        # assert os.path.exists(args.output_dir)
        # args.output_dir = os.path.join(args.output_dir, 'predictions')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'scores.json'), 'w', encoding='utf-8') as fscojson, \
                open(os.path.join(args.output_dir, 'scores.txt'), 'w', encoding='utf-8') as fscotxt:
            json.dump(results, fscojson, ensure_ascii=False, indent=4)
            fscotxt.write("{} {}".format(
                results_score_only[0][0], results_score_only[0][1]))
            for rs in results_score_only[1:]:
                fscotxt.write("\n{} {}".format(rs[0], rs[1]))
            fscotxt.write("\nAccuracy: {}".format(accuracy))
            print("Accuracy: {}".format(accuracy))
