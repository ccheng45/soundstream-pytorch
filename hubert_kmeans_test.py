import torch
from hubert_kmeans import HubertWithKmeans
import torchaudio

hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = 'hubert/hubert_base_ls960_L9_km500.bin'
dataset_folder= 'data/train-clean-100' 

wav2vec = HubertWithKmeans(
    checkpoint_path = hubert_ckpt,
    kmeans_path = hubert_quantizer
).eval()

x, sr = torchaudio.load('temp/LockChime.wav')

print("x", x.shape)
with torch.no_grad():
    output = wav2vec(x)
    print("output", output.shape)