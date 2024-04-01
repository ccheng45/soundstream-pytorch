import torch

from decoder_transformer import DecoderOnlyTransformer

model = DecoderOnlyTransformer()
model.eval()
input = torch.rand(1, 64, 1024)  # batch, len, d
output = model.forward(input)
print(output.shape)
