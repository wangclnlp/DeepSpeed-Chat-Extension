# Step2: Direct Preference Optimization: Your Language Model is Secretly a Reward Model

paper: https://arxiv.org/abs/2305.18290

## Format of the data

The dataset for DPO is the same as that for training reward model, which should be parquet files including `train.parquet` and `test.parquet` as example below.

Example:

| prompt | response | chosen | rejected |
|:---:|:---:|:---:|:---:|
| User: What are some of the challenges with usi... | Some of the challenges with using machine lear... | Some of the challenges with using machine lear... | Machine learning is a very powerful tool. |
| User: Looking for an essay by a contemporary m... | I believe you're thinking of Bernard-Henri Lévy. | I believe you're thinking of Bernard-Henri Lévy. | Laclau maybe? |
| ... | ... | ... | ... |


## How to train the model

```bash
bash examples/dpo/train.sh
```

## Citation

```
@article{rafailov2023direct,
  title={Direct preference optimization: Your language model is secretly a reward model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}
```
