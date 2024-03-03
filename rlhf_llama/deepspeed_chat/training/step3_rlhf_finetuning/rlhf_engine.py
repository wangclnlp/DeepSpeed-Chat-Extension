# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import os
import deepspeed
from deepspeed import get_accelerator
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler

from rlhf_llama.deepspeed_chat.training.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from rlhf_llama.deepspeed_chat.training.utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from rlhf_llama.deepspeed_chat.training.utils.model.model_utils import create_hf_model, create_critic_model
from rlhf_llama.deepspeed_chat.training.utils.utils import get_optimizer_grouped_parameters, print_rank_0
from rlhf_llama.deepspeed_chat.training.utils.ewc import EWC
"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class DeepSpeedRLHFEngine():

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters, reward_model_name_or_path=None, reward_tokenizer=None):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        if reward_model_name_or_path == None:
            print('Using critice_model_name_or_path as reward_model_name_or_path')
            reward_model_name_or_path = critic_model_name_or_path
            
        # if reward_tokenizer == None:
        #     raise "Need reward tokenizer!"
        # else:
        self.reward_tokenizer = reward_tokenizer

        if self.args.iterative_alignment:
            import os
            self.ewc = EWC(args.lamda_factor, args.ewc_max_weight)
            print_rank_0("extract parameters of the previous rlhf model..................................")
            previous_parameters_dict = {}
            if "base" in self.args.previous_sft_model:
                for bin_path in os.listdir(self.args.previous_sft_model):
                    if "0000" in bin_path:
                        previous_parameters_dict_sub = torch.load(self.args.previous_sft_model+"/"+bin_path,  map_location='cpu')
                        previous_parameters_dict.update(previous_parameters_dict_sub)
            else:
                previous_parameters_dict = torch.load(self.args.previous_sft_model+"/pytorch_model.bin",  map_location='cpu')

            print_rank_0("extract parameters of the current model..................................")
            current_model_parameters_dict = {}
            if "base" in actor_model_name_or_path:
                for bin_path in os.listdir(actor_model_name_or_path):
                    if "0000" in bin_path:
                        current_model_parameters_dict_sub = torch.load(actor_model_name_or_path+"/"+bin_path,  map_location='cpu')
                        current_model_parameters_dict.update(current_model_parameters_dict_sub)
            else:
                current_model_parameters_dict = torch.load(actor_model_name_or_path+"/pytorch_model.bin",  map_location='cpu')

        self.actor = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path)
        
        if self.args.iterative_alignment:
            # check if the fisher from the previous round exists.
            import os
            print_rank_0("compute the fisher..................................")
            if os.path.exists(f"{self.args.previous_round_after_sft_model}/previous_fisher.bin"):
                previous_fisher = torch.load(f"{self.args.previous_round_after_sft_model}/previous_fisher.bin")
                self.ewc.fisher = previous_fisher
                # compute the fisher number
                for n in current_model_parameters_dict.keys():
                    fp = current_model_parameters_dict[n.replace("module.", "")]
                    previous_fp = previous_parameters_dict[n.replace("module.", "")]
                    # compute the different between rlhf model and previous rlhf model.
                    self.ewc.fisher[n.replace("module.", "")] += (((fp - previous_fp)*self.args.ewc_mse_factor) ** 2).mean().item()
            else:
                # compute the fisher number
                for n in current_model_parameters_dict.keys():
                    fp = current_model_parameters_dict[n.replace("module.", "")]
                    previous_fp = previous_parameters_dict[n.replace("module.", "")]
                    # compute the different between sft model and previous sft model.
                    # from rlhf_llama.deepspeed_chat.training.utils import pdb ; pdb.set_trace()
                    self.ewc.fisher[n.replace("module.", "")] = (((fp - previous_fp)*self.args.ewc_mse_factor) ** 2).mean().item()

            # <--baseline method(HAT-Freeze): freezing parameters--begin>
            using_hat_freeze = False
            if using_hat_freeze:
                freeze_rate = 0.2
                sorted_fisher = dict(sorted(self.ewc.fisher.items(), key=lambda item: item[1], reverse=True))
                select_freeze_parameter = list(sorted_fisher.keys())[:int(len(sorted_fisher) * freeze_rate)] 

                for n, p in self.actor.named_parameters():
                    if n.replace("module.", "") in select_freeze_parameter:
                        p.requires_grad = False
            # <--baseline method(HAT-Freeze): freezing parameters--end>

            # save the fisher in this round
            torch.save(self.ewc.fisher, f"{actor_model_name_or_path}/previous_fisher.bin")
            # using softmax to normalize the fisher
            norm_values = torch.softmax(torch.Tensor(list(self.ewc.fisher.values())), dim=-1)


            # <--baseline method(HAT-Random): random weights--begin>
            using_hat_random = False
            if using_hat_random:
                random_fisher_value = torch.rand_like(norm_values)
                norm_values = torch.softmax(random_fisher_value, dim=-1)
            # <--baseline method(HAT-Random): random weights--end>

            self.ewc.fisher = dict(zip(self.ewc.fisher.keys(), norm_values.tolist()))

            print_rank_0("end...delete parameters dict..................................")
            del previous_parameters_dict
            self.ewc.reference_model_parameters = current_model_parameters_dict
            # del current_model_parameters_dict
            import gc
            gc.collect()

        if not args.remove_critic_model:
            self.critic = self._init_critic(
                critic_model_name_or_path=critic_model_name_or_path)

        if not args.use_comet_model:
            self.reward = self._init_reward(
                reward_model_name_or_path=reward_model_name_or_path)
        else:
            self.reward = self.load_comet_reward_model(
                self.args.comet_model_path, self.args.devices_comet_model, local_rank=args.local_rank)

        if not self.args.remove_kl_penalty:
            self.ref = self._init_ref(
                actor_model_name_or_path=actor_model_name_or_path)

        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)

        if (not args.remove_critic_model) and self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    def load_comet_reward_model(self, comet_model_path, devices_comet_model, local_rank):
        import os
        from comet import load_from_checkpoint

        # Amazing code for disabling logging output of `pytorch_lightning` while training. 
        # For more details, please turn to https://github.com/Lightning-AI/lightning/issues/3431#issuecomment-915949734 
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        reward_models = []

        # get device
        if local_rank == -1:
            device = torch.device(get_accelerator().device_name())
        else:
            get_accelerator().set_device(local_rank)
            device = torch.device(get_accelerator().device_name(), local_rank)

        # for model_path, device in zip(comet_model_path, devices_comet_model):
        for model_path in comet_model_path:  # single comet model.
            assert (os.path.exists(model_path)), "Invalid path of Comet reward model!"
            reward_model = load_from_checkpoint(model_path)
            reward_model.to(device)
            # reward_model = reward_model.half()
            reward_model.eval()
            reward_models.append(reward_model)

        return reward_models

    def _init_actor(self, actor_model_name_or_path, if_previous_sft_model=False):
        if if_previous_sft_model:
            stime = log_init("Previous SFT Model")
        else:
            stime = log_init("Actor")

        # DS Config
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            # offload=True,
            dtype=self.args.dtype,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=(not self.args.unpin_actor_parameters),
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            max_out_tokens=self.args.max_prompt_seq_len +
            self.args.max_answer_seq_len)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor
        ds_config['deepspeed_multinode_launcher'] = 'standard' 
        ds_config['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
        ds_config['wall_clock_breakdown'] = False

        # Model
        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            disable_dropout=self.args.disable_actor_dropout)
        
        if if_previous_sft_model:
            return actor_model

        # LoRA
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        # AdamOptimizer = DeepSpeedCPUAdam if True else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            actor_model, self.args.actor_weight_decay)
        optim = AdamOptimizer(optim_params,
                              lr=self.args.actor_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim,
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config)

        log_init("Actor", stime=stime)

        return actor_engine

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")
        # DS Config
        # zero_stage = self.args.actor_zero_stage
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
            zero_stage = 0
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       dtype=self.args.dtype,
                                       stage=zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor
        ds_config['deepspeed_multinode_launcher'] = 'standard'
        ds_config['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
        ds_config['wall_clock_breakdown'] = False

        ref_model = create_hf_model(AutoModelForCausalLM,
                                    actor_model_name_or_path, self.tokenizer,
                                    ds_config)

        ref_engine, *_ = deepspeed.initialize(model=ref_model,
                                              config=ds_config)

        log_init("Ref", stime=stime)
        return ref_engine

    def _init_ema(self, actor_model_name_or_path):
        stime = log_init("EMA")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       dtype=self.args.dtype,
                                       stage=zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor
        ds_config['deepspeed_multinode_launcher'] = 'standard' 
        ds_config['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
        ds_config['wall_clock_breakdown'] = False

        actor_model_ema = create_hf_model(AutoModelForCausalLM,
                                          actor_model_name_or_path,
                                          self.tokenizer, ds_config)
        if self.args.actor_lora_dim > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)

        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config)

        log_init("EMA", stime=stime)
        return ema_engine

    def _init_critic(self, critic_model_name_or_path):
        stime = log_init("Critic")
        ds_config = get_train_ds_config(offload=self.args.offload_critic_model,
                                        dtype=self.args.dtype,
                                        stage=self.args.critic_zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps
        ds_config['deepspeed_multinode_launcher'] = 'standard' 
        ds_config['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
        # ds_config['wall_clock_breakdown'] = False

        #However, do not use this ds_eval_config during the init process. 
        ds_eval_config = get_eval_ds_config(offload=False, dtype=self.args.dtype, stage=0)

        # Model 
        critic_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            disable_dropout=self.args.disable_critic_dropout,
            zero_stage=self.args.critic_zero_stage)

        # LoRA
        if self.args.critic_lora_dim > 0:
            critic_model = convert_linear_layer_to_lora(
                critic_model, self.args.critic_lora_module_name,
                self.args.critic_lora_dim)
            if self.args.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload_critic_model else FusedAdam
        optim_pararms = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay)
        optim = AdamOptimizer(optim_pararms,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)

        log_init("Critic", stime=stime)
        return critic_engine

    def _init_reward(self, reward_model_name_or_path):
        stime = log_init("Reward")
        # DS Config
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If reward is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0

        ds_config = get_eval_ds_config(offload=self.args.offload,
                                       dtype=self.args.dtype,
                                       stage=zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        ds_eval_config = get_eval_ds_config(offload=False, stage=0)

        # Model
        reward_model = create_critic_model(
            model_name_or_path=reward_model_name_or_path,
            tokenizer=self.reward_tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            zero_stage=self.args.critic_zero_stage,
            is_reward=True)

        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_config)

        log_init("Reward", stime=stime)
        return reward_engine
