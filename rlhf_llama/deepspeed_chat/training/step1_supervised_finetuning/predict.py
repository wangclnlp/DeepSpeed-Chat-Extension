# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
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

from transformers import LlamaTokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from rlhf_llama.deepspeed_chat.training.utils.model.model_utils import create_hf_model

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to trained model",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output results.",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help='Specify num of beams',
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help='Path to test data.',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help='Batch size to generation.',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=1.0,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             top_k=50,
             top_p=0.95,
             repetition_penalty=1.0,
             num_return_sequences=1,
             max_new_tokens=512):

    gen_kwargs = {
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty":repetition_penalty,
            "num_return_sequences": num_return_sequences,
            "synced_gpus": True,
            "output_scores": True,
            "return_dict_in_generate": True
        }
    
    gen_out = model.module.generate(inputs.input_ids,
                                    attention_mask=inputs.attention_mask,
                                    **gen_kwargs)

    # generate_ids = model.generate(inputs.input_ids,
    #                               num_beams=num_beams,
    #                               num_beam_groups=num_beam_groups,
    #                               do_sample=do_sample,
    #                               top_k=top_k,
    #                               top_p=top_p,
    #                               repetition_penalty=repetition_penalty,
    #                               num_return_sequences=num_return_sequences,
    #                               max_new_tokens=max_new_tokens)

    seq = gen_out["sequences"]
    prompts = inputs.input_ids
    
    batch_size = seq.shape[0]
    prompt_length = prompts.shape[1]
    prompt_length = prompt_length

    out_seq = []
    for i in range(batch_size):
        eos_inds = (seq[i][prompt_length:] == tokenizer.eos_token_id).nonzero()
        eos_ind = eos_inds[0].item() + prompt_length if len(eos_inds) > 0 else max_new_tokens+prompt_length
        seq[i][eos_ind + 1:] = tokenizer.pad_token_id
        
        out_seq.append(seq[i][prompt_length:])

    
    result = tokenizer.batch_decode(out_seq,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)

    print(result)
    
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, model, tokenizer, device,
                all_batch):

    res_writer =  open(args.output_file, "a+", encoding="utf-8")
    for batch in tqdm(all_batch):
        # print("==========finetune: Greedy=========")
        # r_finetune_g = generate(model,
        #                         tokenizer,
        #                         inputs,
        #                         num_beams=1,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_g)
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        res = generate(model, tokenizer, batch,
                                num_beams=1,
                                do_sample=True,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                repetition_penalty=args.penalty_alpha,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        
        for src, pre, ref in zip(batch["source"], res, batch["reference"]):
            src = src.replace("\n", "\\n")
            pre = pre.replace("\n", "\\n")
            ref = ref.replace("\n", "\\n")
            res_writer.write(src+" ||| "+pre+" ||| "+ref+"\n")

        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)


def main():
    args = parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
                                              fast_tokenizer=True)
    
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.add_eos_token_id = False
    tokenizer.padding_side = 'left'    # see rlhf_llama/deepspeed_chat/training/utils/data/data_utils.py  line: 245

    all_batch = []
    batch_size = args.batch_size
    tmp_data = []
    references = []
    sources = []
    text = open(args.test_data, "r", encoding="utf-8")
    for line in text:
        line = line.strip().split(" ||| ")
        references.append(line[1])
        tmp_data.append(line[0])
        sources.append(line[0])
        if len(tmp_data) == batch_size:
            prompts = tokenizer(tmp_data, padding=True, return_tensors="pt").to(device)
            prompts["source"] = sources
            prompts["reference"] = references
            all_batch.append(prompts)
            tmp_data = []
            sources = []
            references = []

    if len(tmp_data) != 0:
        prompts = tokenizer(tmp_data, padding=True, return_tensors="pt").to(device)
        all_batch.append(prompts)


    # load model
    print("load model..............")
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer, None)
    
    # DS Config
    ds_config = get_train_ds_config(
        offload=False,
        dtype="bf16",
        stage=3,
        enable_hybrid_engine=False,
        inference_tp_size=1,
        max_out_tokens=args.max_new_tokens)

    # DeepSpeed Engine
    #TODO: move enable_hybrid_engine and pin_parameters to ds_config
    model, *_ = deepspeed.initialize(model=model,
                                    config=ds_config)
    
    prompt_eval(args, model, tokenizer, device, all_batch)

    # writing results
    # with open(args.output_file, "w", encoding="utf-8") as f:
    #     for src, pre, ref in zip(sources, results, references):
    #         src = src.replace("\n", "\\n")
    #         pre = pre.replace("\n", "\\n")
    #         ref = ref.replace("\n", "\\n")
    #         f.write(src+" ||| "+pre+" ||| "+ref+"\n")

if __name__ == "__main__":
    main()
