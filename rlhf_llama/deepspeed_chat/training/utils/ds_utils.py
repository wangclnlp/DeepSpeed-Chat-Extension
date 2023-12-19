# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        dtype="fp16",
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):

    device = "cpu" if offload else "none"

    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True,
                        "min_loss_scale": 0.5,
                        "fp16_scale_tolerance": 0.25,
                        "loss_scale_window": 1000}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}

    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "contiguous_gradients": False,
        "overlap_comm": False,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_gather_16bit_weights_on_model_save": True,
        "stage3_max_live_parameters": 1e9,
        "stage3_prefetch_bucket_size": 1e9,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_eval_ds_config(offload, dtype="fp16", stage=0):
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True,
                        "min_loss_scale": 0.5,
                        "fp16_scale_tolerance": 0.25,
                        "loss_scale_window": 1000}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}

    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
