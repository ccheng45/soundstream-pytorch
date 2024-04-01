import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from utils import CfgNode as CN

from transformer_einsops import TransformerBlockEnins

# def get_default_config():
#     C = CN()
#     C.model_type = 'gpt'
#     C.n_layer = None
#     C.n_head = None
#     C.n_embd =  None
#     # these options must be filled in externally
#     C.vocab_size = None
#     C.block_size = None
#     # dropout hyperparameters
#     C.embd_pdrop = 0.1
#     C.resid_pdrop = 0.1
#     C.attn_pdrop = 0.1
#     return C


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers=12, dim=1024, heads=16, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlockEnins(dim=dim, heads=heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)

