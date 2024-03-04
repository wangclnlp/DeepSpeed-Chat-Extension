# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
 
# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
from . import raw_datasets
from transformers import DataCollatorForSeq2Seq


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    if "Dahoas/rm-static" in dataset_name: 
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "gpt_annotation_score" in dataset_name: 
        return raw_datasets.GPTAnnotationScoreRMDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "reward_data_neu" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "reward_data_gaobao" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "reward" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "rlhf_gaobao" in dataset_name:
        return raw_datasets.MultiTurnAlpacaDataset(output_path, seed,
                                                   local_rank, dataset_name)
    elif "rlhf_neu" in dataset_name:
        return raw_datasets.MultiTurnAlpacaDataset(output_path, seed,
                                                   local_rank, dataset_name)
    elif "sft" in dataset_name:
        return raw_datasets.MultiTurnSFTDataset(output_path, seed,
                                                   local_rank, dataset_name)
    elif "rlhf" in dataset_name:
        return raw_datasets.MultiTurnRLHFDataset(output_path, seed,
                                                   local_rank, dataset_name)
    elif "MultiTurnAlpaca" in dataset_name:
        return raw_datasets.MultiTurnAlpacaDataset(output_path, seed,
                                                   local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               dataset_name)
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    if "/" in dataset_name:
        dataset_name = dataset_name.replace("/", "_")
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if not os.path.isfile(index_file_name):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase, train_phase3_with_sft_data=False, 
                 gpt_annotated_score_to_train_rm=False,
                 add_error_sample=False) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase
        self.train_phase3_with_sft_data = train_phase3_with_sft_data
        self.gpt_annotated_score_to_train_rm = gpt_annotated_score_to_train_rm
        self.add_error_sample = add_error_sample

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["labels"]
            }
        elif self.train_phase == 2:
            if self.gpt_annotated_score_to_train_rm:
                if self.add_error_sample:
                    return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                        self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"], \
                        self.chosen_dataset[idx]["gpt_score"], self.reject_dataset[idx]["gpt_score"], \
                        self.chosen_dataset[idx]["gpt_correctness"]
                return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                    self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"], \
                    self.chosen_dataset[idx]["gpt_score"], self.reject_dataset[idx]["gpt_score"]
            else:
                return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                    self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
            # return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
            #     self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"], \
            #     5,4
        elif self.train_phase == 3:
            if self.train_phase3_with_sft_data:
                return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                    self.pad_token_id, self.prompt_dataset[idx]["sft_input_ids"], self.prompt_dataset[idx]["sft_attention_mask"], \
                    self.prompt_dataset[idx]["sft_labels"]
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len, train_phase3_with_sft_data=False,
                         input_and_output_max_len=None,
                         args=None):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            """
            "history": [
                ["user instruction in the first round (optional)", "model response in the first round (optional)"],
                ["user instruction in the second round (optional)", "model response in the second round (optional)"]
            ]
            """
            history = raw_dataset.get_history(tmp_data)
            history_token = tokenizer("",return_tensors="pt")
            history_token['input_ids'] = torch.Tensor()
            history_token['attention_mask'] = torch.Tensor()
            history_token['labels'] = torch.Tensor()

            if history != []:
                for item in history:
                    query, response = item[0], item[1]
                    response += end_of_conversation_token

                    query_token = tokenizer(query, return_tensors="pt")
                    response_token = tokenizer(response, return_tensors="pt")

                    input_ids = torch.cat((query_token['input_ids'].squeeze(0), response_token['input_ids'].squeeze(0)[1:]), dim=-1)
                    attention_masks = torch.cat((query_token['attention_mask'].squeeze(0), response_token['attention_mask'].squeeze(0)[1:]), dim=-1)
                    labels = torch.cat((torch.LongTensor([-100]*len(query_token['attention_mask'].squeeze(0))), response_token['input_ids'].squeeze(0)[1:]), dim=-1)

                    history_token['input_ids'] = torch.cat((history_token['input_ids'], input_ids), dim=-1) 
                    history_token['attention_mask'] = torch.cat((history_token['attention_mask'], attention_masks), dim=-1) 
                    history_token['labels'] = torch.cat((history_token['labels'], labels), dim=-1) 



            # tokenize the text
            real_response = raw_dataset.get_chosen(tmp_data) # output 
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # instruction + output 
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         return_tensors="pt")
                
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                
                if len(chosen_token["input_ids"]) + len(history_token['input_ids']) > max_seq_len:
                    continue
                
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                
                if chosen_token["input_ids"][-1] != tokenizer.eos_token_id:
                    chosen_token["input_ids"] = torch.cat((chosen_token["input_ids"], torch.LongTensor([tokenizer.eos_token_id])), dim=-1)
                    chosen_token["attention_mask"] = torch.cat((chosen_token["attention_mask"], torch.LongTensor([1])), dim=-1)

                # creat the labels
                real_response = tokenizer(real_response,
                                         return_tensors="pt")
                real_response["input_ids"] = real_response["input_ids"].squeeze(0)

                if real_response["input_ids"][-1] != tokenizer.eos_token_id:
                    response_length = len(real_response["input_ids"])
                else:
                    response_length = len(real_response["input_ids"])-1

                chosen_token["labels"] = torch.cat((torch.LongTensor([-100]*(len(chosen_token["input_ids"]) - response_length)),
                                                     chosen_token["input_ids"][-response_length:]), dim=-1)

                # chosen_token["labels"] = chosen_token["input_ids"]

                if history != []:
                    chosen_token['input_ids'] = torch.cat((history_token['input_ids'], chosen_token['input_ids']), dim=-1)
                    chosen_token['attention_mask'] = torch.cat((history_token['attention_mask'], chosen_token['attention_mask']), dim=-1)
                    chosen_token['labels'] = torch.cat((history_token['labels'], chosen_token['labels']), dim=-1)

                chosen_dataset.append(chosen_token)

        return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                        tokenizer.pad_token_id, train_phase)

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_token = tokenizer(chosen_sentence,
                                         return_tensors="pt")
                
                if len(chosen_token["input_ids"]) > max_seq_len-1:
                    continue
                
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

                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                
                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                
                # if use gpt-annotated scores to train the reward model.
                if args and args.gpt_annotated_score:
                    chosen_token["gpt_score"] = raw_dataset.get_chosen_gpt_score(tmp_data)
                    reject_token["gpt_score"] = raw_dataset.get_rejected_gpt_score(tmp_data)

                    if args.add_error_sample:
                        chosen_token["gpt_correctness"] = raw_dataset.get_gpt_correctness(tmp_data)

                chosen_dataset.append(chosen_token)
                reject_dataset.append(reject_token)

        if args!= None:
            return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                                tokenizer.pad_token_id, train_phase, train_phase3_with_sft_data=train_phase3_with_sft_data,
                                gpt_annotated_score_to_train_rm=args.gpt_annotated_score, add_error_sample=args.add_error_sample)
        else:
            return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                                tokenizer.pad_token_id, train_phase, train_phase3_with_sft_data=train_phase3_with_sft_data)

    elif train_phase == 3:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                prompt_token["input_ids"] = prompt_token["input_ids"]
                prompt_token["attention_mask"] = prompt_token["attention_mask"]
                falg = False
                for key_word in ["input_ids", "attention_mask"]:
                    length = prompt_token[key_word].size()[-1]
                    if key_word == "input_ids":
                        if length > max_seq_len:
                            falg = True
                            break    # not need when data is enough
                        else:
                            y = prompt_token[key_word].squeeze(0).flip(0)  # why flip? DataCollatorRLHF 用于实现left padding
                        prompt_token[key_word] = y
                    else:
                        y = prompt_token[key_word].squeeze(0).flip(0)  # why flip? DataCollatorRLHF 用于实现left padding
                        prompt_token[key_word] = y
                if falg:
                    continue
                
                ## train phase 3 with sft data
                if train_phase3_with_sft_data:
                    real_chosen_sentence = raw_dataset.get_chosen(tmp_data) 
                    sft_input_and_output = raw_dataset.get_prompt_and_chosen(tmp_data)

                    # += end_of_conversation_token
                    real_chosen_sentence += tokenizer.eos_token
                    sft_input_and_output += tokenizer.eos_token
                    
                    sft_input_and_ouput_token = tokenizer(sft_input_and_output,
                                                            truncation=True,
                                                            return_tensors="pt")

                    if len(sft_input_and_ouput_token["input_ids"].squeeze(0)) > input_and_output_max_len:
                        continue
                    sft_input_and_ouput_token["input_ids"] = sft_input_and_ouput_token["input_ids"].squeeze(0)
                    sft_input_and_ouput_token["attention_mask"] = sft_input_and_ouput_token["attention_mask"].squeeze(0)

                    # real_chosen_sentence += end_of_conversation_token
                    real_chosen_token = tokenizer(real_chosen_sentence,
                                                        truncation=True,
                                                        return_tensors="pt")
                    real_chosen_token["input_ids"] = real_chosen_token["input_ids"].squeeze(0)
                    if len(real_chosen_token["input_ids"]) > input_and_output_max_len-max_seq_len:
                        continue 
                    real_chosen_token["attention_mask"] = real_chosen_token["attention_mask"].squeeze(0)

                    chosen_token_len = len(sft_input_and_ouput_token["input_ids"])
                    prompt_token['sft_input_ids'] = sft_input_and_ouput_token["input_ids"]
                    prompt_token['sft_attention_mask'] = sft_input_and_ouput_token["attention_mask"]

                    response_len = len(real_chosen_token["input_ids"])

                    prompt_token["sft_labels"] = prompt_token['sft_input_ids'].clone()
                    # for label_index in range(len(prompt_token["sft_labels"])):
                    prompt_token["sft_labels"][:-response_len] = 0
                    prompt_token["sft_labels"][-response_len:] = 1
                    
                    # prompt_token["sft_labels"][: chosen_token_len - response_len] = -100
                    ## end
                prompt_dataset.append(prompt_token)
            
        return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                        tokenizer.pad_token_id, train_phase, train_phase3_with_sft_data=train_phase3_with_sft_data)

    


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, train_phase3_with_sft_data=False,
                   input_and_output_max_len=None,
                   args=None):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len, train_phase3_with_sft_data=train_phase3_with_sft_data,
                                         input_and_output_max_len=input_and_output_max_len,
                                         args=args)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len, train_phase3_with_sft_data=train_phase3_with_sft_data,
                                        input_and_output_max_len=input_and_output_max_len,
                                        args=args)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="",
                          sft_only_data_path=[],
                          train_phase3_with_sft_data=False,
                          input_and_output_max_len=None,
                          args=None):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and buf_create_cache.item() != 0:
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_split, output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len, train_phase3_with_sft_data=train_phase3_with_sft_data,
                input_and_output_max_len=input_and_output_max_len, args=args)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank, d_path, data_split, output_path, train_phase,
                    seed, tokenizer, end_of_conversation_token, max_seq_len, 
                    train_phase3_with_sft_data=train_phase3_with_sft_data,
                    input_and_output_max_len=input_and_output_max_len)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                )
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                # train_dataset = ConcatDataset(sft_train_datasets)
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                # eval_dataset = ConcatDataset(sft_eval_datasets)
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    print("wait signal……")
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] +
                                       [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        if len(data[-1]) == 6:
            batch["gpt_score"] = torch.Tensor([f[4] for f in data] +
                                           [f[5] for f in data])
        elif len(data[-1]) == 7:
            batch["gpt_score"] = torch.Tensor([f[4] for f in data] +
                                           [f[5] for f in data])
            batch["gpt_correctness"] = [f[6] for f in data]

        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size, pad_token_id, tokenizer):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.sft_data_collator = DataCollatorForSeq2Seq(
                                tokenizer,
                                pad_to_multiple_of=8,
                                return_tensors="pt",
                                padding=True,
                                label_pad_token_id=0
                            )

    def __call__(self, data):
        batch = {}
        # pad_token_id = data[-1][-1]  # todo: bug, should use the real pad token id
        pad_token_id = self.pad_token_id

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of self.max_token_len (don't use it to reduce memory)
        # length = prompt.size()[-1]
        # pad_length = self.max_token_len - length
        # if pad_length > 0:
        #     batch["prompt"] = F.pad(prompt,
        #                             pad=(0, pad_length),
        #                             mode='constant',
        #                             value=pad_token_id)
        #     batch["prompt_att_mask"] = F.pad(prompt_mask,
        #                                      pad=(0, pad_length),
        #                                      mode='constant',
        #                                      value=0) 
        # else:
        batch["prompt"] = prompt
        batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        if len(data[-1]) < 4:
            pass
        else:
            tmp = [{"input_ids": f[3],
                    "attention_mask": f[4],
                    "labels": f[5]} for f in data]
            ### make sure the final ouput is a seqence of 2**?
            tmp = self.sft_data_collator.__call__(tmp)
            batch["sft_input_ids"] = tmp["input_ids"]
            batch["sft_attention_mask"] = tmp["attention_mask"]
            batch["sft_labels"] = tmp["labels"]
        
        
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset: 
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()
 
        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []
