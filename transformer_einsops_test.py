import torch
from transformer_einsops import SimpleAttentionEins, MultiHeadSelfAttentionEnins

def test_simple():
    model = SimpleAttentionEins(dim=256)
    x = torch.rand(7, 50, 256)
    output = model(x)
    print("Output shape: ", output.shape)

def test_multihead():
    model = MultiHeadSelfAttentionEnins(dim=256)
    x = torch.rand(7, 50, 256)
    output = model(x)
    print("Output shape: ", output.shape)

# test_simple()
test_multihead()
