# Add extra loss

👉 **The ``--add_sft_loss`` argument**

We added this argument for adding SFT loss.

👉 **The ``--add_pretrained_loss`` argument**

We added this argument for adding pretrained loss.

👉 **The ``--factor_rl_loss`` argument**

Loss will be `factor_rl_loss * loss_rl` if `add_sft_loss` or `add_pretrained_loss`

👉 **The ``--factor_sft_loss`` argument**

Loss will be added by `factor_sft_loss * loss_sft` if `add_sft_loss`

👉 **The ``--factor_pretrained_loss`` argument**

Loss will be added by `factor_pretrained_loss * loss_pretrained` if `add_pretrained_loss`

## How to train the model

```bash
bash examples/add_extra_loss/train.sh
```

## Citation

```
@article{wang2023esrl,
  title={ESRL: Efficient Sampling-based Reinforcement Learning for Sequence Generation},
  author={Wang, Chenglong and Zhou, Hang and Hu, Yimin and Huo, Yifu and Li, Bei and Liu, Tongran and Xiao, Tong and Zhu, Jingbo},
  journal={arXiv preprint arXiv:2308.02223},
  year={2023}
}
```
