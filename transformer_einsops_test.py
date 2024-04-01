import torch
from transformer_einsops import SimpleAttentionEins, MultiHeadSelfAttentionEins, TransformerBlockEnins

def test_simple():
    model = SimpleAttentionEins(dim=256)
    x = torch.rand(7, 50, 256)
    output = model(x)
    print("Output shape: ", output.shape)

def test_multihead():
    model = MultiHeadSelfAttentionEins(dim=256)
    x = torch.rand(7, 50, 256)
    output = model(x)
    print("Output shape: ", output.shape)

def test_gpt_like_transformer_with_blocks():
    model =TransformerBlockEnins(dim=256)
    x = torch.rand(7, 50, 256)
    output = model(x)
    print("Output shape: ", output.shape)

# test_simple()
# test_multihead()
test_gpt_like_transformer_with_blocks()