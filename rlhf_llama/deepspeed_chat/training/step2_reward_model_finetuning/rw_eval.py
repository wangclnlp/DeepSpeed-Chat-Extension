#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch

from transformers import AutoTokenizer
import sys
from tqdm import tqdm

from transformers import LlamaTokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from rlhf_llama.deepspeed_chat.training.utils.model.model_utils import create_critic_model
from rlhf_llama.deepspeed_chat.training.utils.utils import to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # default=0,
        help="data path for evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        # default=0,
        help="batch size for evaluation.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="dropout value for MC dropout.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        # default=0,
        help="data path for writting the results.",
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning, dropout=None):
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path,
                                               fast_tokenizer=False)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token = True
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True, dropout=dropout, is_reward=True)

    return model, tokenizer


def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=1024,
                     end_of_conversation_token=None):
    if end_of_conversation_token != None:
        chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
        reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response

    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=1024,
                         end_of_conversation_token=None):
    if end_of_conversation_token != None:
        chosen_sentence = prompt + good_ans + end_of_conversation_token
    
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch

def prepare_singlesample_from_dataset(dataset_path,
                         batch_size,
                         tokenizer,
                         max_seq_len=1024,
                         end_of_conversation_token=None):

    text = open(dataset_path, "r", encoding="utf-8")

    batchs = []
    tmp_batch = []
    for sample_id, sample_line in enumerate(text):
        sample_line = sample_line.strip().split(" ||| ")
        sample_content = f"{sample_line[0]} ||| {sample_line[1]} ||| {sample_line[2]}"
        if len(sample_line) != 3: 
            gpt_label_score = float(sample_line[-1])

        if end_of_conversation_token != None:
            chosen_sentence = sample_line[1] + sample_line[2] + end_of_conversation_token
        else:
            chosen_sentence = sample_line[1] + sample_line[2]

        chosen_token = tokenizer(chosen_sentence,
                                 return_tensors="pt")
                
        if len(chosen_token["input_ids"]) > max_seq_len-1:
            continue
    
        chosen_token = tokenizer(chosen_sentence,
                                max_length=max_seq_len,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")

        batch = {}
        batch["sample_id"] = sample_id
        if len(sample_line) != 3:
            batch["gpt_label_score"] = gpt_label_score
        batch["sample_content"] = sample_content
        batch["input_ids"] = chosen_token["input_ids"]
        batch["attention_mask"] = chosen_token["attention_mask"]
        tmp_batch.append(batch)

        if len(tmp_batch) == batch_size:
            new_batch = {}
            new_batch["input_ids"] = torch.cat([item["input_ids"] for item in tmp_batch],
                                                dim=0)
            new_batch["attention_mask"] = torch.cat([item["attention_mask"] for item in tmp_batch],
                                                dim=0)
            new_batch["sample_id"] = [item["sample_id"] for item in tmp_batch]
            if len(sample_line) != 3:
                new_batch["gpt_label_score"] = [item["gpt_label_score"] for item in tmp_batch]
            new_batch["sample_content"] = [item["sample_content"] for item in tmp_batch]
            batchs.append(new_batch)
            tmp_batch = []
    
    if len(tmp_batch) > 0:
        new_batch = {}
        new_batch["input_ids"] = torch.cat([item["input_ids"] for item in tmp_batch],
                                            dim=0)
        new_batch["attention_mask"] = torch.cat([item["attention_mask"] for item in tmp_batch],
                                            dim=0)
        new_batch["sample_id"] = [item["sample_id"] for item in tmp_batch]
        if len(sample_line) != 3:
            new_batch["gpt_label_score"] = [item["gpt_label_score"] for item in tmp_batch]
        new_batch["sample_content"] = [item["sample_content"] for item in tmp_batch]
        batchs.append(new_batch)

    return batchs


def run_pair_comparison():
    args = parse_args()

    device = torch.device("cuda:0")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning,
                                     dropout=args.dropout)
    rm_model.to(device)
    rm_model.bfloat16()
    rm_model.eval()

    prompt_list = [
        "Human: Please tell me about Microsoft in a few sentence? Assistant: ",
        "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant: "
    ]
    good_ans_list = [
        "Microsoft is a software company that develops, licenses, and supports software products, including Windows, Office, and Windows Phone. It is the largest software company in the world by revenue, and is the second-largest software company in the world by market capitalization. Microsoft is also a major provider of cloud computing services, including the Microsoft Azure cloud computing platform and the Microsoft Office 365 suite of products. The company was founded in 1975",
        "The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another"
    ]
    bad_ans_list = [
        "I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure.",
        "I don't know, I don't know."
    ]

    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(prompt,
                                 good_ans,
                                 bad_ans,
                                 tokenizer,
                                 max_seq_len=1024)
        
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        print("==================Eval result============================")
        print("prompt: ", prompt)
        print("\ngood_ans: ", good_ans)
        print("\nbad_ans:", bad_ans)
        print()
        print("=============Scores (higher, better)========================")
        print("good_ans score: ", outputs["chosen_mean_scores"].item())
        print("bad_ans score: ", outputs["rejected_mean_scores"].item())


def run_single_sample(if_compute_correlation=False):
    args = parse_args()
    device = torch.device("cuda")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning,
                                     args.dropout)
    rm_model.to(device)
    rm_model.bfloat16()
    batchs = prepare_singlesample_from_dataset(args.dataset,
                                batch_size=args.batch_size,
                                tokenizer=tokenizer,
                                max_seq_len=1024)
    all_res = {}
    res_infile = open(args.output_file, "a+", encoding="utf-8")
    remove_colums = ["sample_id", "gpt_label_score", "sample_content"]
    for batch in tqdm(batchs):
        del_unused_colum = {}
        sample_id = batch["sample_id"]
        if "gpt_label_score" in batch.keys():
            gpt_label_score = batch["gpt_label_score"]
        sample_content = batch["sample_content"]
        for item in batch.keys():
            if item not in remove_colums:
                del_unused_colum[item] = batch[item]
        del_unused_colum = to_device(del_unused_colum, device)

        # rm_model.eval()
        # Running inference
        with torch.no_grad():
            outputs = rm_model.forward_value(**del_unused_colum)
            # we just need to skip the number of padding tokens at the beginning
        if "gpt_label_score" in batch.keys():
            for id, pre_score, label_score, content in zip(sample_id, outputs["chosen_end_scores"], gpt_label_score, sample_content):
                all_res[id] = [str(pre_score.item()), str(label_score), content]
        else:
            for id, pre_score, content in zip(sample_id, outputs["chosen_end_scores"], sample_content):
                all_res[id] = [str(pre_score.item()), content]

        # Writing scores to a file
        for sample_id in all_res.keys():
            res_infile.write(str(sample_id) + " ||| " + " ||| ".join(all_res[sample_id]) + "\n")
            
        all_res = {}

    if if_compute_correlation:
    #compute correlation
        pre_scores = []
        label_scores = []

        for sample_id in all_res.keys():
            pre_scores.append(all_res[sample_id][0].item())
            label_scores.append(all_res[sample_id][1])
        
        import numpy as np
        from scipy.stats import spearmanr
        from scipy.stats import kendalltau

        pre_scores = np.array(pre_scores)
        label_scores = np.array(label_scores)
        # 使用numpy的corrcoef函数计算皮尔逊相关系数
        correlation_matrix = np.corrcoef(pre_scores, label_scores)
        pearson_coefficient = correlation_matrix[0, 1]

        # 使用scipy的spearmanr函数计算斯皮尔曼秩相关系数
        spearman_coefficient, _ = spearmanr(pre_scores, label_scores)

        # 使用scipy的kendalltau函数计算肯德尔秩相关系数
        kendall_coefficient, _ = kendalltau(pre_scores, label_scores)
        print("######################################################################")
        print(f"Model: {args.model_name_or_path}\n")
        print(f"pearson_coefficient:{round(pearson_coefficient, 3)}\t\
            spearman_coefficient: {round(spearman_coefficient, 3)}\tkendall_coefficient:{round(kendall_coefficient, 3)}")
        print("######################################################################")
        # write the results
        with open(args.output_file, "a+", encoding="utf-8") as f:
            f.write(f"pearson_coefficient:{round(pearson_coefficient, 3)}\t\
                    spearman_coefficient:{round(spearman_coefficient, 3)}\tkendall_coefficient:{round(kendall_coefficient, 3)}\n")

if __name__ == "__main__":
    # run_pair_comparison()
    run_single_sample()
