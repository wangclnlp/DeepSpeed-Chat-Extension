import argparse
import sys
import os
import json
from dschat.utils.utils import load_hf_tokenizer
import torch
from torch import nn
from tqdm import tqdm
import math
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the reward scores.")
    parser.add_argument(
        "--score_json",
        type=str,
        help="Path to scores.json.",
        required=True,
    )
    parser.add_argument(
        "--sim_method",
        type=str,
        default=None,
        choices=['cosine_similarity', 'length_ratio'],
        help="Method to calculate similarity.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save similarity.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    device = 'cuda:0'
    assert os.path.exists(args.score_json)
    with open(args.score_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    c_scores = [d['c_score'] for d in data]
    r_scores = [d['r_score'] for d in data]
    delta = [c-r for c,r in zip(c_scores, r_scores) if c>r]
    delta_max = max(delta)
    delta_min = min(delta)
    delta_ave = sum(delta)/len(delta)
    print("delta_max: {}, delta_min: {}, delta_ave:{}".format(delta_max, delta_min, delta_ave))

    if args.sim_method != None:
        if args.output_dir == None:
            args.output_dir = os.path.dirname(args.score_json)
        chosen = [d['chosen'] for d in data]
        rejected = [d['rejected'] for d in data]
        c_r = chosen + rejected
        if args.sim_method == 'length_ratio':
            model_name_or_path = os.path.join(os.getenv('DSCHAT'), './models/meta-llama/Llama-2-7b-hf')
            add_eot_token = False
            end_of_conversation_token = "<|endoftext|>"
            additional_special_tokens = end_of_conversation_token if add_eot_token else None
            tokenizer = load_hf_tokenizer(model_name_or_path,
                                        fast_tokenizer=True,
                                        add_special_tokens=additional_special_tokens)
            tokenizer.padding_side = 'right'
            tokenized_seq = tokenizer(c_r,
                                      max_length=args.max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors="pt").input_ids
            print(c_r[0])
            print(tokenized_seq[0])
            data_len = len(tokenized_seq) // 2
            data_similarity = []
            for i in tqdm(range(data_len)):
                chosen_id = tokenized_seq[i]
                rejected_id = tokenized_seq[i+data_len]
                c_inds = (chosen_id == tokenizer.pad_token_id).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else len(chosen_id)
                check_divergence = (chosen_id != rejected_id).nonzero()

                if len(check_divergence) == 0:
                    end_ind = rejected_reward.size(-1)
                    divergence_ind = end_ind - 1
                    r_ind = c_ind
                else:
                    r_inds = (rejected_id == tokenizer.pad_token_id).nonzero()
                    r_ind = r_inds[0].item() if len(r_inds) > 0 else len(rejected_id)
                    end_ind = max(c_ind, r_ind)
                    divergence_ind = check_divergence[0].item()
                assert divergence_ind > 0
                c_truncated_pad_size = len(
                    (chosen_id[divergence_ind:end_ind] == tokenizer.pad_token_id).nonzero())
                r_truncated_pad_size = len(
                    (rejected_id[divergence_ind:end_ind] == tokenizer.pad_token_id).nonzero())
                longer_len = end_ind - divergence_ind
                if longer_len != 0:
                    similarity = (
                        longer_len - max(c_truncated_pad_size, r_truncated_pad_size)) / longer_len
                else:
                    similarity = 1.
                data_similarity.append(similarity.item() if type(similarity)==torch.Tensor else similarity)

        elif args.sim_method == 'cosine_similarity':
            ref_model_name_or_path = os.path.join(os.getenv('DSCHAT'), 'models/google-bert/bert-base-cased')
            ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name_or_path)
            ref_tokenizer.padding_side = 'right'
            ref_model = AutoModel.from_pretrained(ref_model_name_or_path).to(device)
            ref_model.eval()
            batch_size = 16
            chosen = [d['chosen'].replace('</s>', '') for d in data]
            rejected = [d['rejected'].replace('</s>', '') for d in data]
            data_len = len(chosen)
            data_split = [batch_size] * (data_len//batch_size)
            if data_len % batch_size != 0:
                data_split.append(data_len % batch_size)
            ind = 0
            chosen_split, rejected_split = [], []
            data_similarity = []
            print_num = 1
            for i in data_split:
                chosen_split.append(chosen[ind:ind+i])
                rejected_split.append(rejected[ind:ind+i])
                ind += i
            for c,r in tqdm(zip(chosen_split, rejected_split), total=len(chosen_split)):
                d = c + r
                bs = len(c)
                tokenized_seq = ref_tokenizer(d,
                                        max_length=512,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors="pt").to(device)
                while print_num > 0:
                    print(d[0])
                    print(tokenized_seq['input_ids'][0])
                    print_num -= 1
                with torch.no_grad():
                    ref_output = ref_model(**tokenized_seq)
                seq_cls = ref_output[0][:, 0, :]
                seq_cls_nor = nn.functional.normalize(seq_cls, p=2, dim=1)
                for i in range(bs):
                    similarity = torch.clamp(torch.nn.functional.cosine_similarity(
                        seq_cls_nor[i].unsqueeze(0), seq_cls_nor[i+bs].unsqueeze(0)), min=0, max=1)
                    similarity = (math.pi - torch.acos(similarity)) / math.pi
                    data_similarity.append(similarity.item() if type(similarity)==torch.Tensor else similarity)

        data_with_sim = []
        for c, r, c_s, r_s, sim in zip(chosen, rejected, c_scores, r_scores, data_similarity):
            data_with_sim.append({
                'chosen': c,
                'rejected': r,
                'c_score': c_s,
                'r_score': r_s,
                'delta_score': c_s - r_s,
                'similarity': sim
            })
        with open(os.path.join(args.output_dir, 'similarity_{}.json'.format(args.sim_method)), 'w', encoding='utf-8') as f:
            json.dump(data_with_sim, f, ensure_ascii=False, indent=4)
