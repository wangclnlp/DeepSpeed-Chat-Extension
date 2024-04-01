#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import torch

import deepspeed
from dschat.utils.model.model_utils import create_critic_model, create_hf_model
from dschat.utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator
from dschat.utils.ds_utils import get_eval_ds_config

import json
import os
from torch import multiprocessing
from tqdm import tqdm
from transformers import AutoModelForCausalLM


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
        default=None,
        help="Dir of the output of test dataset.",
    )
    # parser.add_argument(
    #     "--num_padding_at_beginning",
    #     type=int,
    #     default=1,
    #     help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
    #     "We did not see this in other models but keep it as an option for now.",
    # )
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
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


def load_stuff(model_name_or_path,
               additional_special_tokens):

    tokenizer = load_hf_tokenizer(model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = 'left'
    ds_config = {
        "fp16": {"enabled": args.dtype=='fp16'},
        "bf16": {"enabled": args.dtype=='bf16'},
        "zero_optimization": {
            "stage": 3,
        },
        "train_micro_batch_size_per_gpu": args.batch_size,
    }
    model = create_hf_model(AutoModelForCausalLM,
                            model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=0.)
    model, *_ = deepspeed.initialize(model=model, config=ds_config)
    model = model.module
    model.eval()

    return model, tokenizer


def process_single_data(d, dataset='alpaca'):
    if dataset == 'alpaca':
        if not d['instruction'].endswith(d['input']):
            prompt = "\n\nHuman: " + \
                d['instruction'] + d['input'] + "\n\nAssistant: "
        else:
            prompt = "\n\nHuman: " + d['instruction'] + "\n\nAssistant: "
        return prompt, d['output']


def tokenize_data(data, tokenizer, args):
    batch_prompt = []
    batch_output = []
    count = 0
    if args.local_rank in [-1, 0]:
        data = tqdm(data)
    for d in data:
        prompt, output = process_single_data(d, dataset=args.dataset)
        batch_prompt.append(prompt)
        batch_output.append(output)
        count += 1
        if count % args.batch_size == 0:
            batch = tokenizer(batch_prompt,
                              padding=True,
                              return_tensors="pt")
            batch_prompt = []
            batch_output = []
            count = 0
            yield batch, batch_prompt, batch_output
    if count != 0:
        batch = tokenizer(batch_prompt,
                          padding=True,
                          return_tensors="pt")
        yield batch, batch_prompt, batch_output


def run(data, device, args):
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    model, tokenizer = load_stuff(args.model_name_or_path,
                                  additional_special_tokens)

    results = []
    # Run inference
    data_gen = tokenize_data(data, tokenizer, args)
    for batch, instructions, outputs in data_gen:
        batch = to_device(batch, device)
        with torch.no_grad():
            generated = model.generate(
                **batch, max_new_tokens=args.max_seq_len)
                # **batch, do_sample=True, top_p=0.95, temperature=0.75, max_length=args.max_seq_len)
        print(batch["input_ids"].shape)
        generated_seq = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(generated_seq)
        for prediction, instruction, output in zip(generated_seq, instructions, outputs):
            results.append({
                'instruction': instruction,
                'prediction': prediction[len(instruction):],
                'output': output
            })
            print(results[-1])
    return results


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.test_json)
    args.output_dir = args.model_name_or_path if args.output_dir == None else args.output_dir
    assert not os.path.exists(os.path.join(args.output_dir, 'predictions'))

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
        assert os.path.exists(args.output_dir)
        args.output_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as fscojson, \
                open(os.path.join(args.output_dir, 'results.txt'), 'w', encoding='utf-8') as fscotxt:
            json.dump(results, fscojson, ensure_ascii=False, indent=4)
            fscotxt.write("{} ||| {} ||| {}".format(
                results[0]['instruction'], results[0]['prediction'], results[0]['output']))
            for rs in results[1:]:
                fscotxt.write("\n{} ||| {} ||| {}".format(
                    rs['instruction'], rs['prediction'], rs['output']))
