import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


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
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head**-0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        # Step 1
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # Step 2
        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be:
        # [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, "b t (d k h) -> k b h t d ", k=3, h=self.heads))

        # Step 3
        # resulted shape will be: [batch, heads, tokens, tokens]
        scaled_dot_prod = (
            torch.einsum("b h i d , b h j d -> b h i j", q, k) * self.scale_factor
        )

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 4. Calc result per batch and per head h
        out = torch.einsum("b h i j , b h j d -> b h i d", attention, v)

        # Step 5. Re-compose: merge heads with dim_head d
        out = rearrange(out, "b h t d -> b t (h d)")

        # Step 6. Apply final linear transformation layer
        return self.W_0(out)


class TransformerBlockEnins(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None, dropout=0.1):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
        """
        super().__init__()
        self.mhsa = MultiHeadSelfAttentionEins(dim=dim, heads=heads, dim_head=dim_head)
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.Linear(4 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
        # return self.norm_2(self.linear(y) + y)
        x = x + self.mhsa(self.ln_1(x), mask)       
        x = x + self.mlpf(self.ln_2(x))