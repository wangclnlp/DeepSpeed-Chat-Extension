# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
import math
import random
from numpy.linalg import norm


# Note that the following code is modified from
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self,
                 base_model,
                 tokenizer,
                 num_padding_at_beginning=0,
                 compute_fp32_loss=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss
        self.tokenizer = tokenizer

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                scale_control_method=None,
                beta1_for_controlling_scale_of_reward=6,
                beta2_for_controlling_scale_of_reward=0.001,
                beta3_for_controlling_scale_of_reward=0,
                random_constraints_range=[1.0, 2.0],
                fixed_constraints=3.0,
                ref_output=None):
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # # Detokenize and tokenize with ref_tokenzier
        # if scale_control_method == 'cosine_similarity':
        #     assert ref_model != None, \
        #         "Reference model is needed for reward controlling method 'cosine_similarity'"
        #     seq = self.tokenizer.batch_decode(
        #         sequences=input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #     seq_enc = ref_tokenizer(seq, padding=True, return_tensors='pt')
        #     seq_enc = seq_enc.to(ref_model.device)
        #     with torch.no_grad():
        #         seq_emb = ref_model(**seq_enc)
        #     seq_cls = seq_emb[0][:, 0, :]
        #     seq_cls_nor = nn.functional.normalize(seq_cls, p=2, dim=1)
        if scale_control_method == 'cosine_similarity':
            with torch.no_grad():
                seq_cls = ref_output[0][:, 0, :]
                seq_cls_nor = nn.functional.normalize(seq_cls, p=2, dim=1)

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0].item()
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  # use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            if self.compute_fp32_loss:
                c_truncated_reward = c_truncated_reward.float()
                r_truncated_reward = r_truncated_reward.float()

            similarity = None
            delta_star = None
            if scale_control_method == 'length_ratio':
                c_truncated_pad_size = len(
                    (chosen_id[divergence_ind:end_ind] == self.PAD_ID).nonzero())
                r_truncated_pad_size = len(
                    (rejected_id[divergence_ind:end_ind] == self.PAD_ID).nonzero())
                longer_len = end_ind - divergence_ind
                if longer_len != 0:
                    similarity = (
                        longer_len - max(c_truncated_pad_size, r_truncated_pad_size)) / longer_len
                else:
                    similarity = 1
            elif scale_control_method == 'cosine_similarity':
                with torch.no_grad():
                    similarity = torch.clamp(torch.nn.functional.cosine_similarity(
                        seq_cls_nor[i].unsqueeze(0), seq_cls_nor[i+bs].unsqueeze(0)), min=0, max=1)
                    similarity = (math.pi - torch.acos(similarity)) / math.pi

            if scale_control_method != None:
                if scale_control_method == 'random':
                    delta_star = random.uniform(random_constraints_range[0], random_constraints_range[1])
                elif scale_control_method == 'fixed':
                    delta_star = fixed_constraints
                else:
                    with torch.no_grad():
                        delta_star = beta1_for_controlling_scale_of_reward / \
                            (similarity+beta2_for_controlling_scale_of_reward) + \
                            beta3_for_controlling_scale_of_reward
                loss_ori = - \
                    torch.nn.functional.logsigmoid(
                        c_truncated_reward - r_truncated_reward).mean()
                loss_con = - torch.nn.functional.logsigmoid(
                    delta_star - c_truncated_reward + r_truncated_reward).mean()
                # print(loss_ori, loss_con)
                loss += (loss_ori + loss_con)
            else:
                loss += - torch.nn.functional.logsigmoid(c_truncated_reward -
                                                         r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
            "similarity": similarity,
            "delta_star": delta_star
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
