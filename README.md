We have edited the code of project [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) to support many new features as shown below.

# Our new features🎉🎉🎉

- Add extra loss for rlhf in step3 like sft loss and pretrained loss.
- Support [DPO](https://arxiv.org/abs/2305.18290) as step2.
- Implement [ESRL](https://arxiv.org/abs/2308.02223) feature to train efficiently in step3.
- Support COMET model(s) as reward model(s) in step3 RLHF.
- Support using scores to train reward models directly.

More details in [./examples](./examples).

# Installation

You can use anaconda/miniconda to install packages needed for this project.

```bash
conda env create -f conda-env.yml
conda activate dschat
pip install -r requirements.txt
```

# Training models

## Step 1 Supervised Fine-tuning (SFT)

```bash
bash scripts/sft.sh
```

## Step 2 Reward Model Fine-tuning

```bash
bash scripts/reward.sh
```

## Step 2 Direct Pereference Optimization (DPO)

```bash
bash examples/dpo/train.sh
```

## Step 3 Reinforcement Learning from Human Feedback (RLHF)

```bash
bash scripts/rlhf.sh
```

# Format of the dataset

## SFT

Dataset for SFT should be `txt` files including `train.txt` and `test.txt`  with `sft` in path such as `/your/path/to/sft_dataset/train.txt`, containing a json string each line as example below.

Example:

```
{"instruction": "User: Your task is to ... \nAssistant: ", "input": "...", "output": "..."}
...
```

## DPO/Reward

Dataset for Reward/DPO should be parquet files including `train.parquet` and `test.parquet` with `reward` in path such as `/your/path/to/reward_dataset/train.parquet`, containing four keys each entry as example below.

Example:

| prompt | response | chosen | rejected |
|:---:|:---:|:---:|:---:|
| User: What are some of the challenges with usi... | Some of the challenges with using machine lear... | Some of the challenges with using machine lear... | Machine learning is a very powerful tool. |
| User: Looking for an essay by a contemporary m... | I believe you're thinking of Bernard-Henri Lévy. | I believe you're thinking of Bernard-Henri Lévy. | Laclau maybe? |
| ... | ... | ... | ... |

## RLHF

Same as SFT, except for `rlhf` in path such as `/your/path/to/rlhf_dataset/train.txt`.

# Last but not least

Thanks to the [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) project and its contributors❤️❤️❤️!
