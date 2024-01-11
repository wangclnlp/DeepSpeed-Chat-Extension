# Learning Evaluation Models from Large Language Models for Sequence Generation (in Step3)

paper: https://arxiv.org/abs/2308.04386

👉 **The ``--use_comet_model`` argument**

We added this argument for using a single/multiple comet model(s) as reward model(s).

👉 **The ``--comet_model_path`` argument**

Path to the comet model. Accepted format: 1) a single comet-model path, 2) multiple models in the form: path1 path2 ...

👉 **The ``--comet_model_batch_size`` argument**

We added this argument for setting the batch size of the comet model.

👉 **The ``--weights_comet_model`` argument**

We added this argument for setting the weight for each comet model.

👉 **The ``--devices_comet_model`` argument**

We added this argument for setting the cuda device for each comet model. Hint: we can set the same device for all comet models.

## How to Train the Model

```bash
bash examples/comet_reward/train.sh
```

## Citation

```
@article{wang2023learning,
  title={Learning Evaluation Models from Large Language Models for Sequence Generation},
  author={Wang, Chenglong and Zhou, Hang and Chang, Kaiyan and Liu, Tongran and Zhang, Chunliang and Du, Quan and Xiao, Tong and Zhu, Jingbo},
  journal={arXiv preprint arXiv:2308.04386},
  year={2023}
}
```
