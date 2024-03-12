from residual_vq import ResidualVectorQuantizer

import torch

import log_config

log_config.setup_logging(True)

"""
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        code_replace_threshold: float = 0.0001,
        eps: float = 1e-10,
"""
rvq = ResidualVectorQuantizer(
    num_quantizers = 4, 
    num_embeddings= 128, 
    embedding_dim= 512,
).eval()

input = torch.rand(1000, 64 , 512) # batch, len, d
quantized, codes, commit_loss = rvq(input)
print("quantized", quantized.shape)
print("code", codes.shape)
print("commit_loss", commit_loss.shape, commit_loss.item())