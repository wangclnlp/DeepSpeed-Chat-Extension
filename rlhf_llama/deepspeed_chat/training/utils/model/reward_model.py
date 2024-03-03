# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
 
## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):
    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, is_reward=False):
        super().__init__()
        self.config = base_model.config
        try:
            self.dropout = self.config.dropout
        except:
            # don't find dropout attribute
            self.dropout = 0.0 

        if self.dropout != 0.0:
            self.Dropout = nn.Dropout(p=self.dropout)

        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim") and is_reward: 
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        elif hasattr(self.config, "word_embed_proj_dim"):
            # use zero matrix for the critic model
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
            nn.init.zeros_(self.v_head.weight)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            if is_reward:
                self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
            else:
                # use zero matrix for the critic model
                self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
                nn.init.zeros_(self.v_head.weight)

        self.rwtranrsformer = base_model
        self.tokenizer = tokenizer
        self.PAD_ID = tokenizer.pad_token_id

        self.loss_function_with_gpt_score = torch.nn.MSELoss()

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()
 
    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                gpt_score=None,
                gpt_correctness=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=use_cache)

        # hidden_states = transformer_outputs[0]
        hidden_states = transformer_outputs.hidden_states[-1]  # for baichuan after sft

        if self.dropout != 0.0:
            hidden_states = self.Dropout(hidden_states)

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

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        nan_num = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            # c_ind = c_inds[self.num_padding_at_beginning].item() if len(
            #     c_inds
            # ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()  # 排除输入相等的位置，也就是loss 对比学习从  assist：这个token 开始做

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
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])
            
            # if self.args.reward_type == "seq":
            if gpt_score != None:
                if gpt_correctness == None:
                    instance_loss = 0
                    chosen_gpt_score = gpt_score[i].type(chosen_reward[c_ind - 1].dtype)
                    rejected_gpt_score = gpt_score[i+bs].type(chosen_reward[c_ind - 1].dtype)
                    instance_loss += self.loss_function_with_gpt_score(chosen_reward[c_ind - 1], chosen_gpt_score)
                    instance_loss += self.loss_function_with_gpt_score(rejected_reward[r_ind - 1], rejected_gpt_score)
                else:
                    # rankloss + 正则化约束(使用gpt分数作为约束)
                    if gpt_correctness[i] == 1:
                        instance_loss = 0
                        chosen_gpt_score = gpt_score[i].type(chosen_reward[c_ind - 1].dtype)
                        rejected_gpt_score = gpt_score[i+bs].type(chosen_reward[c_ind - 1].dtype)
                        instance_loss += self.loss_function_with_gpt_score(chosen_reward[c_ind - 1], chosen_gpt_score)
                        instance_loss += self.loss_function_with_gpt_score(rejected_reward[r_ind - 1], rejected_gpt_score)
                    else:
                        instance_loss = 0
                        # chosen_gpt_score = gpt_score[i].type(chosen_reward[c_ind - 1].dtype)
                        # rejected_gpt_score = gpt_score[i+bs].type(chosen_reward[c_ind - 1].dtype)
                        # instance_loss += self.loss_function_with_gpt_score(chosen_reward[c_ind - 1], chosen_gpt_score)
                        # instance_loss += self.loss_function_with_gpt_score(rejected_reward[r_ind - 1], rejected_gpt_score)
                        instance_loss += -torch.nn.functional.logsigmoid(chosen_reward[c_ind - 1] - rejected_reward[r_ind - 1])
            else:
                instance_loss = -torch.nn.functional.logsigmoid(chosen_reward[c_ind - 1] - rejected_reward[r_ind - 1])
            # elif self.args.reward_type == "lex":
            #     instance_loss = -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()

            # instance_loss = -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
            # Fixing the numerical instability
            # instance_loss = -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
            # instance_loss = -torch.log(torch.sigmoid(chosen_reward[c_ind - 1] - rejected_reward[r_ind - 1]))
            if not torch.isnan(instance_loss):
                loss += instance_loss
            else:
                nan_num += 1
        if gpt_score != None:
            loss = loss / (len(gpt_score)-nan_num)
        else:
            loss = loss / (bs-nan_num)

        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
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

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            # head_mask=head_mask,
            output_hidden_states=True,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        # hidden_states = transformer_outputs.hidden_states
        hidden_states = transformer_outputs.hidden_states[-1]  # for baichuan after sft

        if self.dropout != 0.0:
            hidden_states = self.Dropout(hidden_states)
            
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [prompt, answer, 0, 0, 0, 0] this is normal
            # assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]

            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]
                c_inds = (input_id == self.PAD_ID).nonzero()
                # c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                # c_ind = c_inds[0].item() + prompt_length if len(
                #     c_inds) > 0 else seq_len
                c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            
            
            # chosen_end_scores = chosen_end_scores + bias

            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
