# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from rlhf_llama.deepspeed_chat.training.utils.model.model_utils import create_hf_model
import argparse
import logging
import torch
import sys
import os
import deepspeed
from rlhf_llama.deepspeed_chat.training.utils.ds_utils import get_train_ds_config
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

class Predict:

    def __init__(self, max_new_tokens, test_data, temperature, top_p, batch_size, output_file,
                 top_k=50, repetition_penalty=1.0, num_return_sequences=1, local_rank=-1):
        self.max_new_tokens = max_new_tokens
        self.test_data = test_data
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.output_file = output_file
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.num_return_sequences = num_return_sequences
        self.kwargs_printed = False
        self.local_rank = local_rank

    def generate(self, model, tokenizer, inputs,
                 num_beams=1, num_beam_groups=1,
                 do_sample=False,
                 top_k=50, top_p=0.95,
                 temperature=0.75,
                 repetition_penalty=1.0,
                 num_return_sequences=1,
                 max_new_tokens=512):

        gen_kwargs = {
            # "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            # "num_beams": 1,
            "temperature": temperature,
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "num_return_sequences": num_return_sequences,
            "synced_gpus": True,
            "output_scores": True,
            "return_dict_in_generate": True
        }
        if not self.kwargs_printed:
            print(gen_kwargs)
            self.kwargs_printed = True

        gen_out = model.module.generate(inputs.input_ids,
                                        attention_mask=inputs.attention_mask,
                                        **gen_kwargs)

        seq = gen_out["sequences"]
        prompts = inputs.input_ids

        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        prompt_length = prompt_length

        out_seq = []
        for i in range(batch_size):
            eos_inds = (seq[i][prompt_length:] ==
                        tokenizer.eos_token_id).nonzero()
            eos_ind = eos_inds[0].item(
            ) + prompt_length if len(eos_inds) > 0 else max_new_tokens+prompt_length
            seq[i][eos_ind + 1:] = tokenizer.pad_token_id

            out_seq.append(seq[i][prompt_length:])

        result = tokenizer.batch_decode(out_seq,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)

        # print(result)

        return result

    def print_utils(self, gen_output):
        for i in range(len(gen_output)):
            print()
            print(gen_output[i])
            print()

    def prompt_eval(self, model, tokenizer, device, all_batch):
        res_writer = open(self.output_file, "a+", encoding="utf-8")
        for batch in tqdm(all_batch):
            res = self.generate(model, tokenizer, batch,
                                num_beams=1,
                                do_sample=True,
                                top_k=self.top_k,
                                top_p=self.top_p,
                                temperature=self.temperature,
                                repetition_penalty=self.repetition_penalty,
                                num_return_sequences=self.num_return_sequences,
                                max_new_tokens=self.max_new_tokens)

            if self.local_rank in [-1, 0]:
                for src, pre, ref in zip(batch["source"], res, batch["reference"]):
                    src = src.replace("\n", "\\n")
                    pre = pre.replace("\n", "\\n")
                    ref = ref.replace("\n", "\\n")
                    res_writer.write(src+" ||| "+pre+" ||| "+ref+"\n")

    def predict(self, model, device, tokenizer):
        all_batch = []
        batch_size = self.batch_size
        tmp_data = []
        references = []
        sources = []
        text = open(self.test_data, "r", encoding="utf-8")
        for line in text:
            line = line.strip().split(" ||| ")
            references.append(line[1])
            tmp_data.append(line[0])
            sources.append(line[0])
            if len(tmp_data) == batch_size:
                prompts = tokenizer(tmp_data, padding=True,
                                    return_tensors="pt").to(device)
                prompts["source"] = sources
                prompts["reference"] = references
                all_batch.append(prompts)
                tmp_data = []
                sources = []
                references = []

        if len(tmp_data) != 0:
            prompts = tokenizer(tmp_data, padding=True,
                                return_tensors="pt").to(device)
            prompts["source"] = sources
            prompts["reference"] = references
            all_batch.append(prompts)
        self.prompt_eval(model, tokenizer, device, all_batch)
