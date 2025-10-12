import torch as t
from torch import nn

import numpy as np

def weights_init(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0, std=0.01)
                
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def batch_packer(values: t.Tensor | list[t.Tensor], batch_size: int) -> list[t.Tensor] | list[list[t.Tensor]]:
    if isinstance(values, t.Tensor):
        batch = list(t.utils.data.DataLoader(values, batch_size))
        
    elif isinstance(values, list):
        batch = [list(t.utils.data.DataLoader(value, batch_size)) for value in values]

    return batch

def compute_gae(
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_value: np.ndarray,

        gamma: float = 0.99,
        GAE_lambda: float = 0.95
    ):
    # Just computing of GAE.

    gae = 0
    returns = []
    for step in reversed(range(len(values))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * GAE_lambda * (1 - dones[step]) * gae
        
        returns.insert(0, gae + values[step])

        next_value = values[step]

    return returns

def compute_lengths(tensor: t.Tensor, seq_length: int) -> t.Tensor:
    
    # Compute lengths for unpadding
    lengths = t.split(tensor, seq_length, dim=0)
    lengths = [i.size(0) for i in lengths]
    
    return lengths