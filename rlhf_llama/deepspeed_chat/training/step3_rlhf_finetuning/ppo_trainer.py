# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.nn.utils.rnn import pad_sequence

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from rlhf_llama.deepspeed_chat.training.utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

# from alpaca_farm: https://github.com/tatsu-lab/alpaca_farm/blob/b17d605a360fc2936cbe6ee89e81d7270b440903/src/alpaca_farm/torch_ops.py#L61C3-L61C3
def whiten(values, shift_mean=True, epsilon=1e-8, value_queue=None):
    # assert values.size(0) >= 4, f"Internal error: Minibatch size {values.size(0)} is insufficient for whitening."
    if value_queue != None:
        mean, std = value_queue.mean(), value_queue.std(unbiased=False)
    else:
        mean, std = values.mean(), values.std(unbiased=False)
    whitened = (values - mean) / (std + epsilon)
    if not shift_mean:
        whitened = whitened + mean

    return whitened

class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        if not args.remove_critic_model: self.critic_model = self.rlhf_engine.critic
        if not args.remove_kl_penalty: self.ref_model = self.rlhf_engine.ref
        if args.iterative_alignment: self.ewc = self.rlhf_engine.ewc
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        if not args.use_comet_model:
            self.reward_tokenizer = self.rlhf_engine.reward_tokenizer
        else:
            self.reward_tokenizer = None

        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        # self.end_of_conversation_token_id = self.tokenizer(
        #     args.end_of_conversation_token)['input_ids'][-1]

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 10
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = args.gamma
        self.lam = 0.95

        # some attributes for reward queue approach.
        if args.use_comet_model:
            self.reward_queue = [[] for i in range(len(self.reward_model))]  # [reward_1_queue; reward_2_queue; ...; reward_3_queue]
            self.reward_queue_size = args.reward_queue_size
            self.batch_reward_weight= args.batch_reward_weight
            self.cold_boot_count = 0
            self.cold_boot_mode = True

        # queue for whitening reward and value
        if self.args.whiten_rewards:
            saved_reward_whiten_queue = f"{self.args.previous_round_after_rlhf_model}/previous_reward_whiten_queue.bin"
            if os.path.exists(saved_reward_whiten_queue):
                self.reward_whiten_queue = torch.load(saved_reward_whiten_queue).to(self.reward_model.device)
            else:
                self.reward_whiten_queue = torch.tensor([]).to(self.reward_model.device)
        if self.args.whiten_critic_values:
            saved_advantage_whiten_queue = f"{self.args.previous_round_after_rlhf_model}/previous_advantage_whiten_queue.bin"
            if os.path.exists(saved_advantage_whiten_queue):
                self.advantage_whiten_queue = torch.load(saved_advantage_whiten_queue).to(self.reward_model.device)
            else:
                self.advantage_whiten_queue = torch.tensor([]).to(self.reward_model.device)

    def update_reward_queue(self, rewards, if_return_coefficient_uncertainty=False, batch_level=True):
        # reuse the function from llama efficienct tunning framework
        # rewards [reward_1_score, reward_2_score,..., reward_n_score]  reward_1_score: [0.1, 0.5, ..., 0.8]
        if not if_return_coefficient_uncertainty:
            for reward_index, sub_reward in enumerate(rewards):
                tmp_reward_queue = self.reward_queue[reward_index]
                tmp_reward_queue += sub_reward
                if len(tmp_reward_queue) <= self.reward_queue_size:
                    self.reward_queue[reward_index] = tmp_reward_queue
                else:
                    self.reward_queue[reward_index] = tmp_reward_queue[len(tmp_reward_queue)-self.reward_queue_size:]

        elif batch_level:
            for reward_index, sub_reward in enumerate(rewards):
                tmp_reward_queue = self.reward_queue[reward_index]
                tmp_reward_queue += sub_reward
                if len(tmp_reward_queue) <= self.reward_queue_size:
                    self.reward_queue[reward_index] = tmp_reward_queue
                else:
                    self.reward_queue[reward_index] = tmp_reward_queue[len(tmp_reward_queue)-self.reward_queue_size:]

            if if_return_coefficient_uncertainty:
                # compute the coefficient matrix
                coefficient = [[] for i in range(len(rewards))]
                ret = {}
                for reward_i in range(len(rewards)-1):
                    for reward_j in range(reward_i+1, len(rewards)):
                        coefficient_tmp = np.corrcoef(self.reward_queue[reward_i], self.reward_queue[reward_j])[0, 1]
                        coefficient[reward_i].append(coefficient_tmp)
                        coefficient[reward_j].append(coefficient_tmp)

                    ret[f"reward_{reward_i}"] = [np.mean(coefficient[reward_i]), np.std(self.reward_queue[reward_i])]
                ret[f"reward_{len(rewards)-1}"] = [np.mean(coefficient[-1]), np.std(self.reward_queue[-1])]

                return ret
        else:
            batch_size = len(rewards[-1])
            reward_size = len(rewards)
            ret = {}
            for reward_index in range(reward_size):
                ret[f"reward_{reward_index}"] = []
            for sample_index in range(batch_size):
                for reward_index in range(reward_size):
                    tmp_reward_queue = self.reward_queue[reward_index]
                    tmp_reward_queue.append(rewards[reward_index][sample_index])
                    if len(tmp_reward_queue) <= self.reward_queue_size:
                        self.reward_queue[reward_index] = tmp_reward_queue
                    else:
                        self.reward_queue[reward_index] = tmp_reward_queue[len(tmp_reward_queue)-self.reward_queue_size:]

                # compute the coefficient matrix
                coefficient = [[] for i in range(reward_size)]
                for reward_i in range(len(rewards)-1):
                    for reward_j in range(reward_i+1, len(rewards)):
                        coefficient_tmp = np.corrcoef(self.reward_queue[reward_i], self.reward_queue[reward_j])[0, 1]
                        coefficient[reward_i].append(coefficient_tmp)
                        coefficient[reward_j].append(coefficient_tmp)
                    ret[f"reward_{reward_i}"].append([np.mean(coefficient[reward_i]), np.std(self.reward_queue[reward_i])])
                ret[f"reward_{reward_size-1}"].append([np.mean(coefficient[-1]), np.std(self.reward_queue[-1])])

            return ret

    def _generate_sequence(self, prompts, mask, sft_input=None):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        if self.args.add_sft_loss:
            max_answer_seq_len = max(sft_input["sft_label"].sum(dim=-1)).item()
            max_min_length =  min([int((prompts.shape[1] + max_answer_seq_len)*1.2), max_min_length])

        gen_kwargs = {
            "top_k": 50,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_length": max_min_length,
            "synced_gpus": True,
            "output_scores": True,
            "return_dict_in_generate": True
        }

        with torch.no_grad():
            gen_out = self.actor_model.module.generate(
                                                prompts,
                                                attention_mask=mask,
                                                **gen_kwargs
                                            )
            seq, scores = gen_out["sequences"], gen_out["scores"]

        # Dynamic Sampling
        if self.args.dynamic_sampling:
            scores = torch.stack(scores).transpose(0,1)
            log_probs = torch.nn.functional.softmax(scores, dim=-1).max(dim=-1)[0]
            entropy = 1 + torch.mean(torch.exp(log_probs) * log_probs, dim=-1)

            ELIMINATE_RATIO = 0.3
            drop_num = int(np.ceil(ELIMINATE_RATIO * len(entropy)))
            _, eliminated_index = torch.topk(entropy, k=drop_num)
            eliminated_index = eliminated_index.tolist()
            selected_index = [i for i in range(seq.shape[0]) if i not in eliminated_index]
            seq, prompts, mask = [i[selected_index] for i in (seq, prompts, mask)]

        if self.args.add_sft_loss:
            sft_input_after_drop = {}
            if self.args.dynamic_sampling:
                for key_name in sft_input.keys():
                    sft_input_after_drop[key_name] = sft_input[key_name][selected_index]
            else:
                sft_input_after_drop = sft_input

        # End of Dynamic Sampling

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        out_answer = []
        out_reference_answer = []

        for i in range(batch_size):
            if valid_ans_len[i] < 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                eos_inds = (seq[i][prompt_length:] == self.tokenizer.eos_token_id).nonzero()
                eos_ind = eos_inds[0].item() + prompt_length if len(eos_inds) > 0 else max_min_length
                seq[i][eos_ind + 1:] = self.tokenizer.pad_token_id
                out_seq.append(seq[i:i + 1])
                if self.args.add_sft_loss:
                    out_answer.append(seq[i][prompt_length:])
                    out_reference_answer.append(sft_input_after_drop['sft_input_ids'][i][prompt_length:])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        if self.args.use_comet_model and self.args.add_sft_loss:
            return out_seq , out_reference_answer, out_answer, sft_input_after_drop
        elif self.args.add_sft_loss:
            return out_seq, sft_input_after_drop
        else:
            return out_seq

    def generate_experience(self, prompts, mask, sft_input=None):
        self.eval()
        if self.args.use_comet_model and self.args.add_sft_loss:
            seq, ans, ref, sft_input_after_drop = self._generate_sequence(prompts, mask, sft_input=sft_input)
        elif self.args.add_sft_loss:
            seq, sft_input_after_drop = self._generate_sequence(prompts, mask, sft_input=sft_input)   # seq is prompt+generated sequence: [0,0,0,a,b,c,d,0,0]
        else:
            seq = self._generate_sequence(prompts, mask)
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        # Convert `seq` and `attention_mask` if the tokenizer of the critic model is not the same as the actor one.
        if self.args.use_comet_model:
            assert not self.args.shuffle_reward_score_randomly
            prompts_string = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)  # only question
            ans_string = self.tokenizer.batch_decode(ans, skip_special_tokens=True)  # only answer
            ref_string = self.tokenizer.batch_decode(ref, skip_special_tokens=True)  # only reference

            # concatenate the sampled sequence and the reference sequence.  [sampled sequences, reference sequences]
            # So, when using the reference-based comet model, maybe the score of the reference is equal to 1.
            if self.args.add_sft_loss and self.args.update_reference_with_sampling:
                prompts_string += prompts_string
                ans_string += ref_string
                ref_string += ref_string

            # run our comet group
            rewards = []
            # step 1: preprocess input based the prompts_string and the ans_strining.
            reference_based_input = []
            reference_free_input = []

            for s, r, t in zip(prompts_string, ref_string, ans_string):
                reference_based_input.append({"src": s, "ref":r, "mt": t})

            for s, t in zip(prompts_string, ans_string):
                reference_free_input.append({"src": s, "mt": t})

            # step 2: run forward of comet models to obtain reward scores.
            for comet_model in self.reward_model:
                if self.args.comet_model_require_reference:
                    score = comet_model.predict(reference_based_input, batch_size=self.args.comet_model_batch_size,
                                                 gpus=1, progress_bar = False, num_workers=0, devices=[seq.device.index]).scores
                else:
                    score = comet_model.predict(reference_free_input, batch_size=self.args.comet_model_batch_size,
                                                 gpus=1, progress_bar = False, num_workers=0, devices=[seq.device.index]).scores
                rewards.append(score)

            # # step 3: update the reward queue.
            # if self.cold_boot_count > self.reward_queue_size:
            #     self.cold_boot_mode = False

            # if self.cold_boot_mode:
            #     self.update_reward_queue(rewards, if_return_coefficient_uncertainty=False)
            #     self.cold_boot_count += len(rewards[-1])
            # else:
            #     reward_corr_std = self.update_reward_queue(rewards, if_return_coefficient_uncertainty=True, batch_level=self.batch_reward_weight)
            #     # the content of reward_corr_std is: {'reward_0': [1.0, 0.5962855097896772], 'reward_1': [1.0, 0.5962855097896772]}
            #     # fist of the list is the mean of coefficient of association, range in [-1, 1], the second of the list is the standard deviation of reward queue, range in [0,1]

            # # step 4: using it to show our fancy approaches!
            # if self.cold_boot_mode:
            #     combined_rewards = []
            #     for sample_index in range(len(rewards[-1])):
            #             tmp_reward_score = 0
            #             for reward_index in range(len(rewards)):
            #                 tmp_reward_score += 1/len(rewards)*rewards[reward_index][sample_index]
            #             combined_rewards.append(tmp_reward_score)
            # else:
            #     if self.batch_reward_weight:
            #         weight_reward = [reward_corr_std[reward_id][0]*(1/2-reward_corr_std[reward_id][1]) for reward_id in reward_corr_std.keys()]
            #         combined_rewards = []
            #         for sample_index in range(len(rewards[-1])):
            #             tmp_reward_score = 0
            #             for reward_index in range(len(rewards)):
            #                 tmp_reward_score += weight_reward[reward_index]*rewards[reward_index][sample_index]
            #             combined_rewards.append(tmp_reward_score)

            #     else:
            #         combined_rewards = []
            #         for sample_index in range(len(rewards[-1])):
            #             tmp_reward_score = 0
            #             for reward_index in range(len(rewards)):
            #                 tmp_reward_score += reward_corr_std[f"reward_{reward_index}"][sample_index][0]*(1/2-
            #                                     reward_corr_std[f"reward_{reward_index}"][sample_index][1])*rewards[reward_index][sample_index]
            #             combined_rewards.append(tmp_reward_score)

            reward_score = rewards[0]

            # compute actor logits and critic values for the sampled sequence
            with torch.no_grad():
                output = self.actor_model(seq, attention_mask=attention_mask)
                if not self.args.remove_kl_penalty:
                    output_ref = self.ref_model(seq, attention_mask=attention_mask)

                if self.args.reward_type == "lex":
                    if not self.args.remove_critic_model:
                        values = self.critic_model.forward_value(
                                seq, attention_mask, return_value_only=True).detach()[:, :-1]
                    else:
                        values = None
                else:
                    values = None

        else:
            if self.reward_tokenizer is not None:
                prompts_string = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)

                seq_string = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
                # concatenate the sampled sequence and the reference sequence.  [sampled sequences, reference sequences]
                if self.args.add_sft_loss and self.args.update_reference_with_sampling:
                    ref_string = self.tokenizer.batch_decode(sft_input_after_drop["sft_input_ids"], skip_special_tokens=True)
                    seq_string += ref_string

                reward_seq = self.reward_tokenizer(seq_string, padding=True, return_tensors="pt")["input_ids"]
                reward_attention_mask  = reward_seq.not_equal(self.reward_tokenizer.pad_token_id).long()
                reward_seq = reward_seq.to(seq.device)
                reward_attention_mask = reward_attention_mask.to(attention_mask.device)
            else:
                reward_seq = seq
                reward_attention_mask = attention_mask

            with torch.no_grad():
                output = self.actor_model(seq, attention_mask=attention_mask)
                if not self.args.remove_kl_penalty:
                    output_ref = self.ref_model(seq, attention_mask=attention_mask)
                # compute reward score
                reward_score = self.reward_model.forward_value(    # tensor(s_{1}, s_{2}, ..., s{n}), where n is the number of sequences.
                    reward_seq, reward_attention_mask,
                    prompt_length=self.prompt_length)['chosen_end_scores'].detach()
                if self.args.shuffle_reward_score_randomly:
                    shuffle_index = torch.randperm(len(reward_score))
                    reward_score = reward_score[shuffle_index]

                if self.args.reward_type == "lex":
                    if not self.args.remove_critic_model:
                        values = self.critic_model.forward_value(
                                seq, attention_mask, return_value_only=True).detach()[:, :-1]
                    else:
                        values = None
                else:
                    values = None

        logits = output.logits
        if not self.args.remove_kl_penalty:
            logits_ref = output_ref.logits

        # update the reference
        if self.args.add_sft_loss and self.args.update_reference_with_sampling:
            sequence_num = len(reward_score)
            batch_size = sequence_num // 2
            updated_reference = []

            if self.args.use_comet_model:
                for seq_index in range(batch_size):
                    if sum(reward_score[seq_index]) > sum(reward_score[seq_index+batch_size]): # TODO: replace sum with some fancy approaches
                        updated_reference.append(seq[seq_index])
                    else:
                        updated_reference.append(sft_input["sft_input_ids"][seq_index])
            else:
                for seq_index in range(batch_size):
                    if reward_score[seq_index] > reward_score[seq_index+batch_size]:
                        updated_reference.append(seq[seq_index])
                    else:
                        updated_reference.append(sft_input["sft_input_ids"][seq_index])

            reference_tensor = pad_sequence(updated_reference, padding_value=pad_token_id, batch_first=True)
            reference_tensor_mask = reference_tensor.not_equal(self.tokenizer.pad_token_id).long()

            sft_input_after_drop["sft_input_ids"] = reference_tensor
            sft_input_after_drop["sft_attention_mask"] = reference_tensor_mask
            sft_label = sft_input_after_drop["sft_label"]
            col_num_ref = reference_tensor.size(1)
            col_num_label = sft_label.size(1)
            if col_num_ref > col_num_label:
                sft_input_after_drop["sft_label"] = torch.cat((sft_label, torch.ones((batch_size, col_num_ref-col_num_label)).to(sft_label.device)), dim=-1)
            elif col_num_ref < col_num_label:
                sft_input_after_drop["sft_label"] = sft_label[:,0:col_num_ref]

            reward_score = reward_score[:batch_size]

        #### scale the reward scores with bias
        reward_score = torch.Tensor([float(r) + self.args.reward_bias for r in reward_score]).to(logits.device)

        if not self.args.remove_critic_model:
            ret = {
                'prompts': prompts,
                'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
                'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
                'value': values,
                'rewards': reward_score,
                'input_ids': seq,
                "attention_mask": attention_mask,
            }
        elif not self.args.remove_kl_penalty:
            ret = {
                'prompts': prompts,
                'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
                'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
                'rewards': reward_score,
                'input_ids': seq,
                "attention_mask": attention_mask,
            }
        else:
            ret = {
                'prompts': prompts,
                'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
                'rewards': reward_score,
                'input_ids': seq,
                "attention_mask": attention_mask,
            }

        if self.args.add_sft_loss:
            # ret["sft_input_ids"] = sft_input_after_drop["sft_input_ids"]
            # ret["sft_attention_mask"] = sft_input_after_drop["sft_attention_mask"]
            # ret["sft_label"] = sft_input_after_drop["sft_label"]
            ret["sft_input_ids"] = sft_input["sft_input_ids"]
            ret["sft_attention_mask"] = sft_input["sft_attention_mask"]
            ret["sft_label"] = sft_input["sft_label"]

        if self.args.reward_type == "lex":
            if values != None:
                ret["value"] = values

        return ret

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        """
        reward - kl
        :param prompts:
        :param log_probs:
        :param ref_log_probs:
        :param reward_score:
        :param action_mask:
        :return:
        """
        # Hint:
        # if self.config.kl_penalty == "kl":
        #     return logprob - ref_logprob

        # if self.config.kl_penalty == "abs":
        #     return (logprob - ref_logprob).abs()

        # if self.config.kl_penalty == "mse":
        #     return 0.5 * (logprob - ref_logprob).square()

        if not self.args.remove_kl_penalty:
            # if self.args.reward_type == 'seq':
                # r-kl_value_seq_level
                # kl_divergence_estimate = self.kl_ctl * torch.nn.functional.kl_div(log_probs, torch.exp(ref_log_probs), reduction='none').sum(1)
            # else:
            # r-kl_value_token_level
            # kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
            # kl = (logits.softmax(dim=-1) * (logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1))).sum(dim=-1)
            kl = torch.clamp(log_probs - ref_log_probs, min=0.0)
            kl_divergence_estimate = -self.kl_ctl * kl

            # kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs).abs()
            kl_rewards = kl_divergence_estimate
            kl_distance = kl_divergence_estimate

        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)

        # reward_clip = torch.clamp(reward_score, -5, 17)  # scale reward value  (x-min)/(max-min)

        batch_size = log_probs.shape[0]

        if self.args.reward_type == 'seq':
            if self.args.use_comet_model:
                # r-kl_value_seq_level
                # ret = reward_clip - kl_divergence_estimate.view(-1, 1)
                # return ret

                # r-kl_value_token_level (add reward score at last token)
                for j in range(batch_size):
                    kl_rewards[j, start:ends[j]][-1] += reward_clip[j]
            else:
                add_last_token = False
                if not add_last_token:
                    # r-kl_value_token_level
                    ret = []
                    for i in range(batch_size):
                        seq_reward = reward_clip[i]
                        for j in range(len(kl_rewards[i])):
                            seq_reward *= self.gamma
                            kl_rewards[i][j] += seq_reward
                        ret.append(kl_rewards[i].tolist())
                    return torch.Tensor(ret).to(reward_clip.device), torch.abs(kl_distance).mean()
                else:
                    # r-kl_value_token_level (add reward score at last token)
                    for j in range(batch_size):
                        kl_rewards[j, start:ends[j]][-1] += reward_clip[j]
                    return kl_rewards, torch.abs(kl_distance).mean()
        else:
            for j in range(batch_size):
                kl_rewards[j, start:ends[j]][-1] += reward_clip[j]

        return kl_rewards, torch.abs(kl_distance).mean()

    def train_rlhf(self, inputs, update_critic_only=False):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        if not self.args.remove_kl_penalty:
            ref_log_probs = inputs['ref_logprobs']
        else:
            ref_log_probs = None

        reward_score = inputs['rewards']
        if self.args.reward_type == "lex":
            values = inputs['value']

        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        if self.args.reward_type == "lex":
            old_values = values
        else:
            old_values = None

        with torch.no_grad():
            # [reward_1_score; reward_2_score; ...], reward_1_score: [example_1; example_2]\
            # for example (reward_1 and reward_2 are the same):
            '''
            tensor([[-2.0295, -2.0295],
                    [-1.9734, -1.9734]], device='cuda:0')
            '''
            old_rewards, kl_distance = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)

            ends = start + action_mask[:, start:].sum(1) + 1

            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0

                if not self.args.remove_critic_model:
                    old_values[i, ends[i]:] = 0

            if not self.args.remove_critic_model:
                advantages, returns = self.get_advantages_and_returns(
                    old_values, old_rewards, start)
            else:
                advantages = self.get_advantages_and_returns(
                    old_values, old_rewards, start, reward_type='seq')

        all_actor_loss = []
        all_critic_loss  = []
        if update_critic_only:
            batch = {'input_ids': seq, "attention_mask": attention_mask}
            assert not self.args.remove_critic_model, "If you want to update critic only, DONOT remove critic model please."
            value = self.critic_model.forward_value(**batch,
                                            return_value_only=True,
                                            use_cache=False)[:, :-1]
            critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,start:],
                                            returns, action_mask[:, start:])
            self.critic_model.backward(critic_loss)
            self.critic_model.step()
            all_critic_loss.append(critic_loss.item())
            return torch.tensor(0), np.mean(all_critic_loss), old_rewards

        for _ in range(self.args.ppo_mini_epochs):
            self.actor_model.zero_grad()

            ### process the new outputs
            batch = {'input_ids': seq, "attention_mask": attention_mask}

            # update critic model
            if not self.args.remove_critic_model:
                value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]

                critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,start:],
                                                returns, action_mask[:, start:])
                self.critic_model.backward(critic_loss)
                all_critic_loss.append(critic_loss.item())

            # update the actor model
            actor_prob = self.actor_model(**batch, use_cache=False).logits
            actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])


            actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                            log_probs[:, start:], advantages,
                                            action_mask[:, start:])

            ### add sft loss ###
            if self.args.add_sft_loss or self.args.add_pretrained_loss:
                actor_loss = self.args.factor_rl_loss*actor_loss
                self.actor_model.backward(actor_loss, retain_graph=True)
            elif self.args.iterative_alignment:
                self.actor_model.backward(actor_loss, retain_graph=True)
            else:
                self.actor_model.backward(actor_loss)
            logging_loss = actor_loss.item()

            if self.args.add_sft_loss:
                sft_input_ids = inputs["sft_input_ids"]
                sft_attention_mask = inputs["sft_attention_mask"]
                sft_label = inputs["sft_label"]
                batch = {'input_ids': sft_input_ids, "attention_mask": sft_attention_mask}
                sft_prob = self.actor_model(**batch, use_cache=False).logits
                sft_log_prob = - gather_log_probs(sft_prob[:, :-1, :], sft_input_ids[:, 1:])
                sft_loss = torch.sum(sft_log_prob * sft_label[:, 1:]) / torch.sum(sft_label[:, 1:])
                sft_loss = self.args.factor_sft_loss*sft_loss
                self.actor_model.backward(sft_loss)
                logging_loss += sft_loss.item()

            if self.args.add_pretrained_loss:
                pretrained_label = inputs['attention_mask'].clone()
                prompt_length = [prompt.shape[0]-(prompt!=self.tokenizer.pad_token_id).nonzero()[0][-1] for prompt in inputs['prompts']]
                for i, length in enumerate(prompt_length):
                    start_of_prompt = (pretrained_label[i]!=self.tokenizer.pad_token_id).nonzero()[0][-1]
                    end_of_prompt = start_of_prompt + length
                    pretrained_label[i][end_of_prompt:] = 0
                pretrained_loss = - torch.sum(actor_log_prob * pretrained_label[:, 1:]) / torch.sum(pretrained_label[:, 1:])
                pretrained_loss = self.args.factor_pretrained_loss * pretrained_loss
                self.actor_model.backward(pretrained_loss)
                logging_loss += pretrained_loss.item()

            if self.args.iterative_alignment:
                # compute EWC loss for actor loss
                ewc_loss = self.ewc.compute_ewc_loss(self.actor_model)
                self.actor_model.backward(ewc_loss)
                logging_loss += ewc_loss.item()

            # self.actor_model.backward(actor_loss)
            all_actor_loss.append(logging_loss)

            if self.args.align_overflow:
                actor_overflow = self.actor_model.optimizer.check_overflow(
                    external=True)

                if not self.args.remove_critic_model:
                    critic_overflow = self.critic_model.optimizer.check_overflow(
                        external=True)

                rank = torch.distributed.get_rank()
                if not self.args.remove_critic_model:
                    if actor_overflow and not critic_overflow:
                        self.critic_model.optimizer.skip_step = True
                        print_rank_0(
                            "OVERFLOW: actor overflow, skipping both actor and critic steps",
                            rank)
                    elif not actor_overflow and critic_overflow:
                        self.actor_model.optimizer.skip_step = True
                        print_rank_0(
                            "OVERFLOW: critic overflow, skipping both actor and critic steps",
                            rank)
                    elif actor_overflow and critic_overflow:
                        print_rank_0(
                            "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                            rank)
                else:
                    if actor_overflow:
                        print_rank_0(
                            "OVERFLOW: overflow, skipping steps",
                            rank)

            if not self.args.remove_critic_model:
                self.critic_model.step()

            self.actor_model.step()

        if not self.args.remove_critic_model:
            return np.mean(all_actor_loss), np.mean(all_critic_loss), kl_distance
        else:
            return np.mean(all_actor_loss), torch.tensor(0), kl_distance

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        if self.args.use_comet_model:
            # compute loss value with a reward group
            log_ratio = (logprobs - old_logprobs) * mask
            ratio = torch.exp(log_ratio)

            # simple approach: sum(r1+r2+...+rn)*p
            pg_loss = torch.sum(advantages, dim=-1).view(-1, 1) * ratio
            pg_loss = torch.sum(pg_loss * mask) / mask.sum()

            # slove unclear reward problem: sum(r1*p + r2*p + ... + rn*p).
            # Note: maybe it requires more footprints.  a trick: backward, similar to gradient accumulation
            '''
            model_number = advantages.size(1)
            pg_loss = 0
            for i in range(model_number):
                pg_loss += torch.sum(advantages[:,i].view(-1, 1) * ratio, mask) / mask.sum()
            '''

            # slove optimization conflict problem:

        elif self.args.remove_critic_model:
            # log_ratio = (logprobs - old_logprobs) * mask
            ratio = logprobs
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                                1.0 + self.cliprange)
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()

        else:
            log_ratio = (logprobs - old_logprobs) * mask
            ratio = torch.exp(log_ratio)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                                1.0 + self.cliprange)
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()

        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start, reward_type=None):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        if self.args.whiten_rewards:
            tmp_input_queue = rewards[:, start:].reshape(-1)[rewards[:, start:].reshape(-1) != 0]
            self.reward_whiten_queue = torch.cat((self.reward_whiten_queue, tmp_input_queue))[-self.args.reward_queue_size:]
            rewards = whiten(rewards, shift_mean=True, value_queue=self.reward_whiten_queue)

        if reward_type == None:
            lastgaelam = 0
            advantages_reversed = []
            length = rewards.size()[-1]
            for t in reversed(range(start, length)):
                nextvalues = values[:, t + 1] if t < length - 1 else 0.0
                delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
                lastgaelam = delta + self.gamma * self.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            returns = advantages + values[:, start:]
            if self.args.whiten_critic_values:
                tmp_input_queue = advantages[:, start:].reshape(-1)[advantages[:, start:].reshape(-1) != 0]
                self.advantage_whiten_queue = torch.cat((self.advantage_whiten_queue, tmp_input_queue))[-self.args.reward_queue_size:]
                advantages = whiten(advantages, shift_mean=True, value_queue=self.advantage_whiten_queue)
            return advantages.detach(), returns
        else:
            advantages_reversed = []
            length = rewards.size()[-1]
            for t in reversed(range(start, length)):
                advantages_reversed.append(rewards[:, t])
            advantages = torch.stack(advantages_reversed[::-1], dim=1)

            return advantages.detach()

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        if not self.args.remove_critic_model:
            assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        if not self.args.remove_critic_model:
            assert not self.critic_model.module.training
        if not self.args.remove_kl_penalty:
            assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        if not self.args.remove_critic_model:
            self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        if not self.args.remove_critic_model:
            self.critic_model.eval()
        if self.args.use_comet_model:
            pass
        else:
            self.reward_model.eval()
        if not self.args.remove_kl_penalty:
            self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        if not self.args.remove_kl_penalty:
            ref_model_norm = get_model_norm(self.ref_model)
        if not self.args.remove_critic_model:
            critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        if not self.args.remove_kl_penalty:
            print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        if not self.args.remove_critic_model:
            print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
