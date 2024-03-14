from einops import rearrange, repeat
from packaging import version
from collections import namedtuple

from beartype.typing import Optional, Union, List
from beartype import beartype

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum

from func_utils import default, exists, print_once

# Config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        num_null_kv=0,
        dropout=0.1,
        scale=8,
        flash=False,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.num_null_kv = num_null_kv
        self.null_kv = (
            nn.Parameter(torch.randn(2, num_null_kv, dim_head))
            if num_null_kv > 0
            else None
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)

        self.attend = Attend(flash=flash, dropout=dropout, causal=causal)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        prefix_context=None,
        prefix_context_mask=None,
        return_kv_cache=False,
        kv_cache=None,
    ):
        b, n, _, device = *x.shape, x.device

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim=-2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones((b, n), device=device, dtype=torch.bool)

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim=-1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value=True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value=0.0)

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        # kv cache

        if exists(kv_cache):
            ck, cv = kv_cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # store kv cache

        if return_kv_cache:
            kv_cache = torch.stack((k, v))

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, "kv n d -> kv b n d", b=b).unbind(
                dim=0
            )
            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)

        # split for multi-headed attention

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        # handle mask and null key / value

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)

        # attention

        out = self.attend(q, k, v, attn_bias=attn_bias, mask=mask)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, kv_cache


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_context=None,
        cross_attend=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
        grad_shrink_alpha=0.1,
        cond_as_self_attn_prefix=False,
        rel_pos_bias=True,
        flash_attn=False,
        **kwargs,
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        assert not (cross_attend and cond_as_self_attn_prefix)

        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha=grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        self.rel_pos_bias = (
            RelativePositionBias(dim=dim // 2, heads=heads) if rel_pos_bias else None
        )

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            flash=flash_attn,
                            causal=True,
                            **kwargs,
                        ),
                        (
                            Attention(
                                dim=dim,
                                heads=heads,
                                dropout=attn_dropout,
                                dim_context=dim_context,
                                flash=flash_attn,
                                num_null_kv=1,
                                norm_context=True,
                                **kwargs,
                            )
                            if cross_attend
                            else None
                        ),
                        FeedForward(dim=dim, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        self_attn_mask=None,
        context=None,
        context_mask=None,
        attn_bias=None,
        return_kv_cache=False,
        kv_cache=None,
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (
            exists(context) and context.shape[-1] != self.dim_context
        ), f"you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}"

        n, device = x.shape[1], x.device

        # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability

        x = self.grad_shrink(x)

        # turn off kv cache if using conditioning as self attention (as in valle), for now

        if self.cond_as_self_attn_prefix:
            kv_cache = None

        # handle kv cache

        new_kv_cache = []

        if exists(kv_cache):
            cache_len = kv_cache.shape[-2]
            kv_cache = iter(kv_cache)
        else:
            cache_len = 0
            kv_cache = iter([])

        x = x[:, cache_len:]

        # relative positional bias

        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = maybe(self.rel_pos_bias)(n, n)

        if exists(rel_pos_bias):
            rel_pos_bias = rel_pos_bias[..., cache_len:, :]

        # self attention kwargs

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context=context, prefix_context_mask=context_mask
            )

        # transformer layers

        for attn, cross_attn, ff in self.layers:

            residual = x

            x, layer_kv_cache = attn(
                x,
                attn_bias=rel_pos_bias,
                mask=self_attn_mask,
                kv_cache=next(kv_cache, None),
                return_kv_cache=True,
                **self_attn_kwargs,
            )
            new_kv_cache.append(layer_kv_cache)

            x = x + residual

            if exists(cross_attn):
                assert exists(context)

                x = cross_attn(x, context=context, mask=context_mask) + x

            x = ff(x) + x

        x = self.norm(x)

        if not return_kv_cache:
            return x

        return x, torch.stack(new_kv_cache)
