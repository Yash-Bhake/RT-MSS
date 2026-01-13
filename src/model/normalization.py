import torch
import torch.nn as nn


class CumulativeLayerNorm(nn.Module):
    """Cumulative Layer Normalization for causal processing."""
    
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x, running_mean=None, running_var=None, running_count=None):
        """
        Args:
            x: (B, C, T) or (B, T, C)
            running_mean: (B, C) running mean
            running_var: (B, C) running variance
            running_count: (B,) running count
        Returns:
            normalized: normalized output
            new_running_mean: updated running mean
            new_running_var: updated running variance
            new_running_count: updated running count
        """
        # Assume input is (B, T, C) for temporal processing
        if x.dim() == 3 and x.size(1) != self.dim:
            # (B, T, C)
            B, T, C = x.shape
            
            if running_mean is None:
                # Initialize
                running_mean = torch.zeros(B, C, device=x.device, dtype=x.dtype)
                running_var = torch.zeros(B, C, device=x.device, dtype=x.dtype)
                running_count = torch.zeros(B, device=x.device, dtype=x.dtype)
            
            # Update statistics
            batch_mean = x.mean(dim=1)  # (B, C)
            batch_var = x.var(dim=1, unbiased=False)  # (B, C)
            
            new_count = running_count + T
            new_mean = (running_mean * running_count.unsqueeze(1) + batch_mean * T) / new_count.unsqueeze(1)
            
            # Update variance using parallel algorithm
            m_a = running_var * running_count.unsqueeze(1)
            m_b = batch_var * T
            M2 = m_a + m_b + (running_mean - batch_mean).pow(2) * running_count.unsqueeze(1) * T / new_count.unsqueeze(1)
            new_var = M2 / new_count.unsqueeze(1)
            
            # Normalize
            normalized = (x - new_mean.unsqueeze(1)) / (new_var.unsqueeze(1) + self.eps).sqrt()
            normalized = normalized * self.gamma + self.beta
            
            return normalized, new_mean, new_var, new_count
        else:
            # Static normalization for non-temporal dims
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            normalized = (x - mean) / (var + self.eps).sqrt()
            normalized = normalized * self.gamma + self.beta
            return normalized, None, None, None


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
        """
        Args:
            x: (B, C, *) input
        """
        shape = x.shape
        x = x.view(shape[0], self.num_groups, -1)
        
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        
        x = x.view(shape)
        
        # Broadcast gamma and beta
        dims = [1] * len(shape)
        dims[1] = self.num_channels
        gamma = self.gamma.view(*dims)
        beta = self.beta.view(*dims)
        
        return x * gamma + beta