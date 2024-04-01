import torch
from transformers import AutoModelForCausalLM
from torch import multiprocessing
import argparse
import os
import json
from dschat.utils.utils import to_device, load_hf_tokenizer
from dschat.utils.model.model_utils import create_critic_model, create_hf_model
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
        default=None,
        help="Dir of the output of test dataset.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="The temperature set while generating.",
    )
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
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
    args = parser.parse_args()
    return args


def process_single_data(d, dataset='alpaca'):
    if dataset == 'alpaca':
        if not d['instruction'].endswith(d['input']):
            prompt = "\n\nHuman: " + \
                d['instruction'] + d['input'] + "\n\nAssistant: "
        else:
            prompt = "\n\nHuman: " + d['instruction'] + "\n\nAssistant: "
        return prompt, d['output']
    if dataset == 'tldr':
        prompt = "\n\nHuman: Your task is to generate a short summary of a post.\n\nPost: "\
              f"{d['post']}\n\nSummary: \n\nAssistant: "
        return prompt, d['summary']


def tokenize_data(data, tokenizer, args, numprocs):
    batch_prompt = []
    batch_output = []
    count = 0
    if numprocs == 0:
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
            _batch_prompt = batch_prompt
            _batch_output = batch_output
            batch_prompt = []
            batch_output = []
            count = 0
            yield batch, _batch_prompt, _batch_output
    if count != 0:
        batch = tokenizer(batch_prompt,
                          padding=True,
                          return_tensors="pt")
        yield batch, batch_prompt, batch_output


def worker(numprocs, args, data, tokenizer, model, return_dict):
    device = torch.device('cuda:{}'.format(numprocs))
    model.to(device)
    data_gen = tokenize_data(data, tokenizer, args, numprocs)
    results = []
    for batch, instructions, outputs in data_gen:
        batch = to_device(batch, device)
        with torch.no_grad():
            if args.temperature > 0:
                generated = model.generate(
                    **batch, do_sample=True, top_p=0.95, temperature=args.temperature, max_new_tokens=args.max_seq_len)
            else:
                generated = model.generate(
                    **batch, do_sample=False, max_new_tokens=args.max_seq_len)
        # print(batch["input_ids"].shape)
        generated_seq = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(generated_seq)
        for prediction, instruction, output in zip(generated_seq, instructions, outputs):
            results.append({
                'instruction': instruction,
                'prediction': prediction[len(instruction):],
                'output': output
            })
            if numprocs==0:
                print(results[-1])
    print("ID {}: {}".format(numprocs, len(results)))
    return_dict[numprocs] = results


if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.test_json)
    args.output_dir = os.path.join(args.model_name_or_path, 'predictions') if args.output_dir == None else args.output_dir
    assert not os.path.exists(args.output_dir)
    print(args.output_dir)

    num_gpus = torch.cuda.device_count()

    if args.test_json.endswith('json'):
        with open(args.test_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif args.test_json.endswith('jsonl'):
        with open(args.test_json, 'r', encoding='utf-8') as f:
            data = [json.loads(l) for l in f.readlines()]

    data_split = []
    ptr = 0
    len_split = len(data) // num_gpus
    for i in range(num_gpus-1):
        data_split.append(data[ptr:ptr+len_split])
        ptr += len_split
    data_split.append(data[ptr:])

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = 'left'
    # print(tokenizer.encode("<|endoftext|>"))
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer, None)
    if args.dtype == 'fp16':
        model_dtype = torch.float16
    elif args.dtype == 'bf16':
        model_dtype = torch.bfloat16
    model.to(model_dtype)
    print("Init finished!")

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=worker, args=(
            i, args, data_split[i], tokenizer, model, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    data_gathered = []
    for i in range(num_gpus):
        data_gathered += return_dict[i]

    os.makedirs(args.output_dir)
    assert os.path.exists(args.output_dir)
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as fscojson, \
            open(os.path.join(args.output_dir, 'results.txt'), 'w', encoding='utf-8') as fscotxt:
        json.dump(data_gathered, fscojson, ensure_ascii=False, indent=4)
        fscotxt.write("{} ||| {} ||| {}".format(
            data_gathered[0]['instruction'].replace('\n', '\\n'),
            data_gathered[0]['prediction'].replace('\n', '\\n'),
            data_gathered[0]['output'].replace('\n', '\\n')))
        for rs in data_gathered[1:]:
            fscotxt.write("\n{} ||| {} ||| {}".format(
                rs['instruction'].replace('\n', '\\n'),
                rs['prediction'].replace('\n', '\\n'),
                rs['output'].replace('\n', '\\n')))
