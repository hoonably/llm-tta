# lora/dynamic_lora.py
import torch
import torch.nn as nn
import math


class DynamicLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r_max, lora_alpha=16, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.lora_alpha = lora_alpha

        # LoRA low-rank matrices (not used until rank > 0)
        self.lora_A = nn.Parameter(torch.randn(r_max, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, r_max) * 0.01)

        # Scaling factor
        self.scaling = self.lora_alpha / self.r_max

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x, rank=None):
        if rank is None:
            rank = getattr(self, "rank", self.r_max)

        result = nn.functional.linear(x, self.weight, self.bias)
        if rank > 0:
            lora_A_r = self.lora_A[:rank, :]
            lora_B_r = self.lora_B[:, :rank]
            delta = self.scaling * (lora_B_r @ lora_A_r)
            result += nn.functional.linear(x, delta)
        return result
