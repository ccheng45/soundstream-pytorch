from soundstream import StreamableModel, Encoder, Decoder, ResidualVectorQuantizer
import torch
import torch.nn as nn
import torchaudio
    
model = StreamableModel.load_from_checkpoint("temp/epoch=118-step=212000.ckpt")
model.eval()
x, sr = torchaudio.load('temp/LockChime.wav')
# x, sr = torchaudio.load('temp/input.flac')
x, sr = torchaudio.functional.resample(x, sr, 16000), 16000
torchaudio.save('temp/input_downstream.wav', x, sr)
with torch.no_grad():
    # print("x", x.shape)
    x = torch.FloatTensor(x)
    x = x.to("cuda")
    y_hat = model(x)
    y_hat = y_hat[0]
    # print(y_hat.shape)
    # y_hat = y_hat.squeeze(0)
    # print(y_hat.shape)
    y_hat = y_hat.to("cpu").detach()
    torchaudio.save('temp/output.wav', y_hat, sr)