from soundstream import StreamableModel, Encoder, Decoder, ResidualVectorQuantizer
import torch
import torch.nn as nn
import torchaudio

# class EncoderDecoder(nn.Module):
#     def __init__(
#         self,
#         n_channels: int = 32,
#         num_quantizers: int = 8,
#         num_embeddings: int = 1024,
#         padding: str = "same"
#     ):
#         super().__init__()
#         self.encoder = Encoder(n_channels, padding)
#         self.decoder = Decoder(n_channels, padding)
#         self.quantizer = ResidualVectorQuantizer(
#             num_quantizers, num_embeddings, n_channels * 16)

#     def forward(self, x):
#         return self.encode(x)


#     def encode(self, input: torch.Tensor) -> torch.Tensor:
#         print("encode!")
#         assert input.ndim == 2
#         x = torch.unsqueeze(input, 1)
#         x = self.encoder(x)
#         x = torch.transpose(x, -1, -2)
#         _, codes, _ = self.quantizer(x)
#         return codes

#     def decode(self, input: torch.Tensor) -> torch.Tensor:
#         print("decode!")
#         # input: [batch_size, length, num_quantizers]
#         x = self.quantizer.dequantize(input)
#         x = torch.transpose(x, -1, -2)
#         x = self.decoder(x)
        
#         x = torch.squeeze(x, 1)
#         return x
    
model = StreamableModel.load_from_checkpoint("results/epoch=0-step=200.ckpt")
model.eval()
x, sr = torchaudio.load('temp/input.flac')
x, sr = torchaudio.functional.resample(x, sr, 16000), 16000
with torch.no_grad():
    print("x", x.shape)
    x = torch.FloatTensor(x)
    x = x.to("cuda")
    y_hat = model(x)
    y_hat = y_hat[0]
    print(y_hat.shape)
    # y_hat = y_hat.squeeze(0)
    # print(y_hat.shape)
    y_hat = y_hat.to("cpu").detach()
    torchaudio.save('temp/output.wav', y_hat, sr)