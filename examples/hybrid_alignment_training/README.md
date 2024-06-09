# Hybrid Alignment Training for Large Language Models

paper: ***arxiv link***

## Abstract

Alignment training is crucial for enabling large language models (LLMs) to cater to human intentions and preferences. It is typically performed based on two stages with different objectives: instruction-following alignment and human-preference alignment. However, aligning LLMs with these objectives in sequence suffers from an inherent problem: the objectives may conflict, and the LLMs cannot guarantee to simultaneously align with the instructions and human preferences well. To response to these, in this work, we propose a Hybrid Alignment Training (HBAT) approach, based on alternating alignment and modified elastic weight consolidation methods. The basic idea is to alternate between different objectives during alignment training, so that better collaboration can be achieved between the two alignment tasks.  We experiment with HBAT on summarization and dialogue tasks. Experimental results show that HBAT can outperform all baselines. Notably, HBAT yields consistent performance gains over the traditional two-stage alignment training when using both proximal policy optimization and direct preference optimization.

## An Example Script for Training

```bash
# replace with your path at the head of script
bash examples/hybrid_alignment_training/train.sh
```

## Usage of Modified Elastic Weight Consolidation (EWC)

Import EWC first.
```python
from rlhf_llama.deepspeed_chat.training.utils.ewc import EWC
ewc = EWC(args.lamda_factor, args.ewc_max_weight)
```

Compute Fisher before training.
```python
# compute the fisher number
for n in current_model_parameters_dict.keys():
    fp = current_model_parameters_dict[n.replace("module.", "")]
    previous_fp = previous_parameters_dict[n.replace("module.", "")]
    # compute the different between sft model and previous sft model.
    ewc.fisher[n.replace("module.", "")] = (((fp - previous_fp)*args.ewc_mse_factor) ** 2).mean().item()
```

Update Fisher during training.
```python
# Calculate fisher matrix
model.train()
for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
    # ... backward model
    model.backward(logps, retain_graph=True)
    ewc.update_fisher_matrix_with_grad(model)

model.eval()
```

Compute EWC loss and backward.
```python
# add parameter contraint and compute EWC loss for actor loss
ewc_loss = ewc.compute_ewc_loss(model, apply_original_ewc=args.apply_original_ewc)
if ewc_loss != 0:
    model.backward(ewc_loss)
    loss += ewc_loss.item()
```