import torch.nn as nn
import torch
import deepspeed
from rlhf_llama.deepspeed_chat.training.utils.utils import to_device

class EWC():
    def __init__(self, lamda_factor, max_weight=10, train_dataloader=None):
        self.ewc_lambda = lamda_factor

        self.max_weight = max_weight

        self.scheduling_method = "softmax"  # rank

        self.fisher = {}
        self.fisher_matrix = {}
        self.reference_model_parameters = {}

        self.train_dataloader = train_dataloader # the origial ewc requires the grads of LLM on the training set.


    def compute_ewc_loss(self, model, apply_original_ewc=False):
        ewc_loss = 0

        for n, p in model.named_parameters():
            ref_p = self.reference_model_parameters[n.replace("module.", "")]
            if not torch.is_tensor(ref_p):
                ref_p = torch.Tensor(ref_p)

            # solve it with GatheredParameters function.
            # with deepspeed.zero.GatheredParameters(p, modifier_rank=0):
            if p.numel() and p.requires_grad:
                # from rlhf_llama.deepspeed_chat.training.utils import pdb ; pdb.set_trace()
                # ewc_losses.append((self.fisher[n] * (p - ref_p.to(p.device)).pow(2)).sum())
                if apply_original_ewc:
                    ewc_loss += (self.fisher_matrix[n.replace("module.", "")] * self.max_weight * ((p - ref_p.to(p.device))*self.ewc_lambda).pow(2)).sum()
                else:
                    ewc_loss += (self.fisher[n.replace("module.", "")] * self.max_weight * ((p - ref_p.to(p.device))*self.ewc_lambda).pow(2)).sum()
                # from rlhf_llama.deepspeed_chat.training.utils import pdb ; pdb.set_trace()

        return ewc_loss

    def update_fisher_matrix_with_grad(self, model):
        for n, p in model.named_parameters():
            if p.numel() and p.requires_grad:
                n = n.replace("module.", "")
                if n not in self.fisher_matrix.keys():
                    if p.grad:
                        self.fisher_matrix[n] = p.grad.detach().clone()
                    else:
                        self.fisher_matrix[n] = 0
                else:
                    if p.grad:
                        self.fisher_matrix[n] += p.grad.detach().clone()
