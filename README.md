We have edited the code of project [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) to support many new features as shown below.

# Our New Features🎉🎉🎉
- We propose a hybrid alignment training to improve the LLM ([./examples/hybrid_alignment_training](https://github.com/wangclnlp/DeepSpeed-Chat-Extension/tree/main/examples/hybrid_alignment_training)).
- Add extra loss for RLHF in step3 like SFT loss and pre-trained loss ([./examples/add_extra_loss_for_rlhf](./examples/add_extra_loss_for_rlhf)).
- Support [DPO](https://arxiv.org/abs/2305.18290) as step2 ([./examples/dpo](./examples/dpo)).
- Implement [ESRL](https://arxiv.org/abs/2308.02223) features to train efficiently in step3 ([./examples/esrl](./examples/esrl)).
- Support COMET model(s) as reward model(s) in step3 RLHF ([./examples/rlhf_with_comet_reward](./examples/rlhf_with_comet_reward)).
- Support using scores instead of pairwise data only to train reward models directly ([./examples/training_reward_with_scores](./examples/training_reward_with_scores)).

More details in [./examples](./examples).

# Installation

You can use anaconda/miniconda to install packages needed for this project.

```bash
conda env create -f conda-env.yml
conda activate dschat
pip install -r requirements.txt
```

# Training Models

## Step1 Supervised Fine-tuning (SFT)

```bash
bash scripts/sft.sh
```

## Step2 Reward Model Fine-tuning

```bash
bash scripts/reward.sh
```

## Step2 Direct Pereference Optimization (DPO)

```bash
bash examples/dpo/train.sh
```

## Step3 Reinforcement Learning from Human Feedback (RLHF)

```bash
bash scripts/rlhf.sh
```

# Supported Models

| Model | Model size |
|:---:|:---:|
| Baichuan | 7B/13B |
| Baichuan2 | 7B/13B |
| LLaMA | 7B/13B/33B/65B |
| LLaMA-2 | 7B/13B/70B |
| Yi | 6B/34B |

# Format of the Dataset

## SFT

The dataset for SFT should be `txt` files including `train.txt` and `test.txt`  with `sft` in path such as `/your/path/to/sft_dataset/train.txt`, containing a json string each line as example below.

Example:

```
{"instruction": "User: Your task is to ... \nAssistant: ", "input": "...", "output": "..."}
...
```

## SFT with Multi-turn History

We also support sft training with multi-turn dialogues. The corresponding dataset also contains a json string on each line, as shown in the example below.

Example:

```
{
 "instruction": "User: Your task is to ... \nAssistant: ",
 "input": "...",
 "output": "...",
 "history": [
              ["user instruction in the first round (optional)", "model response in the first round (optional)"],
              ["user instruction in the second round (optional)", "model response in the second round (optional)"],
              ...
            ]
}
...
```

## Reward/DPO

The dataset for Reward/DPO should be parquet files including `train.parquet` and `test.parquet` with `reward` in path such as `/your/path/to/reward_dataset/train.parquet`, containing four keys each entry as example below.

Example:

| prompt | response | chosen | rejected |
|:---:|:---:|:---:|:---:|
| User: What are some of the challenges with usi... | Some of the challenges with using machine lear... | Some of the challenges with using machine lear... | Machine learning is a very powerful tool. |
| User: Looking for an essay by a contemporary m... | I believe you're thinking of Bernard-Henri Lévy. | I believe you're thinking of Bernard-Henri Lévy. | Laclau maybe? |
| ... | ... | ... | ... |

## RLHF

Same as SFT, except for `rlhf` in path such as `/your/path/to/rlhf_dataset/train.txt`.

# Inference

You can use [this](rlhf_llama/deepspeed_chat/training/step1_supervised_finetuning/predict.py) python script for inference as shown in [`./scripts/predict.sh`](./scripts/predict.sh) in which the input should be in format of `{Input} ||| {None/Reference}` while output would be `{Input} ||| {ModelOutput} ||| {None/Reference}` as example below.

Example:

input.txt
```
User: What are the names of some famous actors ...\nAssistant: ||| Some famous ...
User: ...                                                      ||| None
...                                                            ||| ...
```

output.txt
```
User: What are the names of some famous actors ...\nAssistant: ||| 1. Denzel Washington ... ||| Some famous ...
User: ...                                                      ||| ...                      ||| None
...                                                            ||| ...                      ||| ...
```


# Last but Not Least

Thanks to the [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) project and its contributors❤️❤️❤️!
