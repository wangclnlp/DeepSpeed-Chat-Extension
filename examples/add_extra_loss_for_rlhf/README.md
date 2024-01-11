# Add Extra Loss for RLHF (in Step3)

👉 **The ``--add_sft_loss`` argument**

Add SFT loss, loss will add `factor_sft_loss * loss_sft`, which means `loss += factor_sft_loss * loss_sft`.

👉 **The ``--add_pretrained_loss`` argument**

Add pre-trained loss, loss will add `factor_pretrained_loss * loss_pretrained`, which means `loss += factor_pretrained_loss * loss_pretrained`.

👉 **The ``--factor_rl_loss`` argument**

Defaults to 0.7, `rl_loss` will be multipied by `factor_rl_loss` if `add_sft_loss` or `add_pretrained_loss`, which means `loss = factor_rl_loss*rl_loss + factor_sft_loss*loss_sft(when add_sft_loss) + factor_pretrained_loss*loss_pretrained(when add_pretrained_loss)`.

👉 **The ``--factor_sft_loss`` argument**

Defaults to 0.15.

👉 **The ``--factor_pretrained_loss`` argument**

Defaults to 0.15.

## How to Train the Model

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
