import torch.nn as nn
import torch
import deepspeed

class EWC():
    def __init__(self, lamda_factor, max_weight=10):
        self.ewc_lambda = lamda_factor

        self.max_weight = max_weight

        self.scheduling_method = "softmax"  # rank

        self.fisher = {}  
        self.reference_model_parameters = {}


    def compute_ewc_loss(self, model):
        ewc_loss = 0

        for n, p in model.named_parameters():
            ref_p = self.reference_model_parameters[n.replace("module.", "")]
            if not torch.is_tensor(ref_p):
                ref_p = torch.Tensor(ref_p)
                
            # solve it with GatheredParameters function.
            with deepspeed.zero.GatheredParameters(p, fwd_module=model, modifier_rank=0):
                if p.requires_grad:
                    # ewc_losses.append((self.fisher[n] * (p - ref_p.to(p.device)).pow(2)).sum())
                    ewc_loss += (self.fisher[n.replace("module.", "")] * self.max_weight * (p - ref_p.to(p.device)).pow(2)).sum()*self.ewc_lambda

        return ewc_loss