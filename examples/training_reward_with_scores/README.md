# Training Reward Model with Scores (in Step2)

👉 **The ``--disable_dropout`` argument**

Disable the dropout of the model.

👉 **The ``--gpt_annotated_score`` argument**

Use GPT-annotated scores to train the reward model.

## Format of the Data

The dataset for training reward model should be parquet files including `train.parquet` and `test.parquet`.

To train reward model with scores, there should be extra keys of `chosen_gpt_score` and `rejected_gpt_score` as example below.

Example:

| prompt | response | chosen | chosen_gpt_score | rejected | rejected_gpt_score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| User: What are some of the challenges with usi... | Some of the challenges with using machine lear... | Some of the challenges with using machine lear... | 0.9 | Machine learning is a very powerful tool. | 0.2 |
| User: Looking for an essay by a contemporary m... | I believe you're thinking of Bernard-Henri Lévy. | I believe you're thinking of Bernard-Henri Lévy. | 0.8 | Laclau maybe? | 0.2 |
| ... | ... | ... | ... | ... | ... |

## How to Train the Model

```bash
bash examples/training_reward_with_scores/train.sh
```
