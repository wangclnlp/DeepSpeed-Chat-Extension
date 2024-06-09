import torch.nn as nn
import torch

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

            if p.numel() and p.requires_grad:
                # ewc_losses.append((self.fisher[n] * (p - ref_p.to(p.device)).pow(2)).sum())
                if apply_original_ewc:
                    ewc_loss += (self.fisher_matrix[n.replace("module.", "")] * self.max_weight * ((p - ref_p.to(p.device))*self.ewc_lambda).pow(2)).sum()
                else:
                    ewc_loss += (self.fisher[n.replace("module.", "")] * self.max_weight * ((p - ref_p.to(p.device))*self.ewc_lambda).pow(2)).sum()

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
