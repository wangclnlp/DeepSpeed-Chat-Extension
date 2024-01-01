
import argparse
import deepspeed
from transformers import SchedulerType
 

def parse_args():
    parser = argparse.ArgumentParser(description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='4,4,2',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='data_cache/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--reward_tokenizer_name_or_path",
        type=str,
        help=
        "Path to reward pretrained model or model identifier from huggingface.co/models.",
        required=False)
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_mini_train_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batch_numbers",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument(
        "--ppo_mini_epochs",
        type=int,
        default=4,
        help="For each sample, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=5,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=-1,
                        help="Save checkpoint in steps.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument('--dtype',
                    type=str,
                    default='fp16',
                    choices=['fp16', 'bf16'],
                    help='Training data type')
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--offload_critic_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for critic and reward model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    parser.add_argument('--disable_actor_dropout',
                        action='store_true',
                        help='Disable the dropout of the actor model.')
    parser.add_argument('--disable_critic_dropout',
                        action='store_true',
                        help='Disable the dropout of the critical model.')
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')
    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')
    ## Remove critic model
    parser.add_argument('--remove_critic_model',
                        action='store_true',
                        help='Remove critic model.')
    ## Save critic model
    parser.add_argument('--save_critic_model',
                        action='store_true',
                        help='Save checkpoints of critic model.')
    ## Train critic model first for n steps.
    parser.add_argument('--critic_go_first_steps',
                        type=int,
                        default=0,
                        help='Train critic model first for n steps.')
    ## Use sequence level reward or lexical level reward
    parser.add_argument('--reward_type',
                        type=str,
                        default='seq',
                        choices=['seq', 'lex'],
                        help='Use sequence level reward or lexical level reward.')
    parser.add_argument('--shuffle_reward_score_randomly',
                        action='store_true',
                        help='Shuffle reward scores randomly.')
    ## Remove KL penalty
    parser.add_argument('--remove_kl_penalty',
                        action='store_true',
                        help='Remove KL penalty.')

    ## Dynamic Sampling
    parser.add_argument('--dynamic_sampling',
                        action='store_true',
                        help='Enable Dynamic Sampling.')

    ## Add SFT loss
    parser.add_argument('--add_sft_loss',
                        action='store_true',
                        help='Add SFT loss.')
    parser.add_argument('--add_pretrained_loss',
                        action='store_true',
                        help='Add pretrained loss.')
    parser.add_argument('--update_reference_with_sampling',
                        action='store_true',
                        help='update the reference with a sampled sequence.')

    ## Factor of sft loss and rl loss
    parser.add_argument('--factor_rl_loss',
                        type=float,
                        default=0,
                        help='factor_rl_loss*loss_rl if add_sft_loss or add_pretrained_loss')
    parser.add_argument('--factor_sft_loss',
                        type=float,
                        default=0,
                        help='+= factor_sft_loss*loss_sft if add_sft_loss')
    parser.add_argument('--factor_pretrained_loss',
                        type=float,
                        default=0,
                        help='+= factor_pretrained_loss*loss_pretrained if add_pretrained_loss')

    ## use the annotated scores to training the reward model. Note: argment confiting, don't set it!
    parser.add_argument('--gpt_annotated_score',
                        type=bool,
                        default=False,
                        help='Use GPT-annotated scores to train the reward model.')

    ## control the reward scale (preventing excessive negative numbers).  
    parser.add_argument('--reward_bias',
                        type=float,
                        default=0.0,
                        help='reward score + bias')
    parser.add_argument('--whiten_rewards',
                        type=bool,
                        default=True,
                        help='Standardizing reward scores within a batch.')



    ## Use comet models to reward LLM; Note that it is a list type
    parser.add_argument('--use_comet_model',
                        action='store_true',
                        help='use a single/multiple comet model(s) as reward model(s).')
    parser.add_argument('--comet_model_path',
                        nargs='*',
                        help='Path to the comet model. Accepted format: 1) a single comet-model path, 2) multiple models in the form: path1 path2 ...')
    parser.add_argument('--comet_model_batch_size',
                        type=int,
                        default=8,
                        help='Path to the comet model. Accepted format: 1) a single comet-model path, 2) multiple models in the form: path1 path2 ...')
    parser.add_argument('--weights_comet_model',
                        nargs='*',
                        help='set a weight for each comet model.')
    parser.add_argument('--devices_comet_model',
                        nargs='*',
                        help='set a cuda device for each comet model. Hint: we can set same device for all comet models.')
    parser.add_argument('--reward_queue_size',
                        type=int,
                        default=20,
                        help='set the size of reward queue.')
    parser.add_argument('--batch_reward_weight',
                        type=bool,
                        default=True,
                        help='if use the batch reward weight; otherwrise, use the sample level reward weight.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if (args.actor_gradient_checkpointing
            and args.actor_lora_dim > 0) or (args.critic_gradient_checkpointing
                                             and args.critic_lora_dim > 0):
        assert (
            not args.only_optimize_lora
        ), "--{actor,critic}_gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args
