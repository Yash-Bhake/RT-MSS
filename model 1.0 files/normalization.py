import torch
import torch.nn as nn

import torch
import torch.nn as nn
from tqdm import tqdm

class CumulativeLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x, running_mean=None, running_var=None, running_count=None):
        """
        Vectorized Causal Normalization for both training and chunked inference.
        """
        B, T, C = x.shape
        device = x.device

        # Initialize states if not provided (Training/First Chunk)
        if running_mean is None:
            running_mean = torch.zeros(B, C, device=device)
            running_var = torch.zeros(B, C, device=device)
            running_count = torch.zeros(B, 1, device=device)

        # 1. Compute cumulative sums for the current chunk
        # t_counts: [1, 2, ..., T]
        t_counts = torch.arange(1, T + 1, device=device).view(1, T, 1).float()
        curr_counts = running_count.view(B, 1, 1) + t_counts
        
        # We need Sum and Sum-of-Squares to update mean/var
        # E[x] = Sum / Count
        curr_sum = torch.cumsum(x, dim=1) + (running_mean * running_count).view(B, 1, C)
        means = curr_sum / curr_counts
        
        # Var[x] = E[x^2] - (E[x])^2
        # E[x^2] = (SumSquares_prev + cumsum(x^2)) / Count_curr
        prev_sum_sq = (running_var + running_mean**2) * running_count
        curr_sum_sq = torch.cumsum(x**2, dim=1) + prev_sum_sq.view(B, 1, C)
        vars = (curr_sum_sq / curr_counts) - (means**2)
        vars = torch.clamp(vars, min=0.0) # Numerical stability
        
        # 2. Normalize and apply affine transform
        x_norm = (x - means) / (vars + self.eps).sqrt()
        out = x_norm * self.gamma + self.beta
        
        # 3. Return last step stats for the next chunk
        return out, means[:, -1, :], vars[:, -1, :], curr_counts[:, -1, 0:1].view(B, 1)

class GroupNorm(nn.Module):
    """Group Normalization for band-level processing."""
    def __init__(self, num_groups, num_channels, eps=1e-8):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, K, N, T = shape
            x = x.reshape(B, self.num_groups, -1)
        else:
            x = x.reshape(shape[0], self.num_groups, -1)
        
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        
        x = x.reshape(shape)
        dims = [1] * len(shape)
        dims[1] = self.num_channels
        gamma = self.gamma.view(*dims)
        beta = self.beta.view(*dims)
        return x * gamma + beta