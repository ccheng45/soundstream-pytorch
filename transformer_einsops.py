import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SimpleAttentionEins(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        self.scale_factor = dim**-0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3, "3D tensor must be provided"

        qkv = self.to_qvk(x)  # [batch, tokens, dim*3 ]

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, "b t (d k) -> k b t d ", k=3))

        # Step 2
        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = (
            torch.einsum("b i d , b j d -> b i j", q, k) * self.scale_factor
        )

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 3
        return torch.einsum("b i j , b j d -> b i d", attention, v)

class MultiHeadSelfAttentionEins(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, max_seq_len=512):
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.relative_position_bias = nn.Parameter(torch.zeros((2 * max_seq_len - 1, heads)))
        
        # Create causal mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        assert x.dim() == 3
        qkv = self.to_qvk(x)
        q, k, v = tuple(rearrange(qkv, "b t (d k h) -> k b h t d ", k=3, h=self.heads))

        scaled_dot_prod = torch.einsum("b h i d , b h j d -> b h i j", q, k) * self.scale_factor

        seq_len = x.size(1)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        scaled_dot_prod = scaled_dot_prod.masked_fill(causal_mask == 0, float("-inf"))

        # Calculate relative positions
        context_position = torch.arange(seq_len, dtype=torch.long, device=x.device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=x.device)[None, :]
        relative_position = memory_position - context_position + seq_len - 1  # Shift range to [0, 2*(seq_len-1)]
        
        relative_position_bias = self.relative_position_bias[relative_position.view(-1)].view(seq_len, seq_len, -1).permute(2, 0, 1)
        scaled_dot_prod += relative_position_bias.unsqueeze(0)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        out = torch.einsum("b h i j , b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.W_0(out)


class TransformerBlockEnins(nn.Module):
    def __init__(self, dim, heads=8 ,dropout=0.1):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dropout: probability of droppping values
        """
        super().__init__()
        self.mhsa = MultiHeadSelfAttentionEins(dim=dim, heads=heads, dim_head=dim//heads)
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.Linear(4 * dim, dim),
            NewGELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.mhsa(self.ln_1(x))       
        x = x + self.linear(self.ln_2(x))
        return x