#!/usr/bin/env python3
"""Check encoder output difference."""

import torch
import numpy as np
import wave
import hashlib
import os

from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as f:
    raw_audio = np.frombuffer(f.readframes(-1), dtype='int16')
speech = raw_audio.astype(np.float32) / 32768.0

from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming as ESPnetStreaming
from speechcatcher.speechcatcher import load_model, tags

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_s2t = ESPnetStreaming(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,
    ctc_weight=0.3,
)
espnet_s2t.reset()

our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()

chunk_size = 8000
for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    espnet_s2t(chunk, is_final=False)
    our_s2t(chunk, is_final=False)

espnet_enc = espnet_s2t.beam_search.encbuffer[:40].unsqueeze(0)
our_enc = our_s2t.beam_search.encoder_buffer[:, :40, :]

print("Encoder output comparison (40 frames):")
print(f"ESPnet shape: {espnet_enc.shape}")
print(f"Ours shape:   {our_enc.shape}")
print(f"\nMax diff: {(espnet_enc - our_enc).abs().max().item():.10f}")
print(f"Mean diff: {(espnet_enc - our_enc).abs().mean().item():.10f}")
print(f"\nClose (atol=1e-5): {torch.allclose(espnet_enc, our_enc, atol=1e-5)}")
print(f"Close (atol=1e-4): {torch.allclose(espnet_enc, our_enc, atol=1e-4)}")
print(f"Close (atol=1e-3): {torch.allclose(espnet_enc, our_enc, atol=1e-3)}")
