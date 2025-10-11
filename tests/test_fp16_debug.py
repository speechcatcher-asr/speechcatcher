#!/usr/bin/env python3
"""Debug FP16 vs FP32 outputs."""

import torch
import numpy as np
import wave
import hashlib
import os

from speechcatcher.speechcatcher import convert_inputfile, load_model, tags

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as f:
    raw_audio = np.frombuffer(f.readframes(-1), dtype='int16')
speech = raw_audio.astype(np.float32) / 32768.0

print("="*80)
print("FP32 (baseline)")
print("="*80)
s2t_fp32 = load_model(tags['de_streaming_transformer_xl'], device='cuda', beam_size=5, quiet=True, fp16=False)
s2t_fp32.reset()

chunk_size = 8000
for i in range(len(speech) // chunk_size + 1):
    chunk = speech[i*chunk_size:min((i+1)*chunk_size, len(speech))]
    if len(chunk) == 0:
        break
    result = s2t_fp32(chunk, is_final=(i == len(speech) // chunk_size))
    if result and result[0][0]:
        print(f"Chunk {i+1}: '{result[0][0]}'")

print("\n" + "="*80)
print("FP16 (testing)")
print("="*80)
s2t_fp16 = load_model(tags['de_streaming_transformer_xl'], device='cuda', beam_size=5, quiet=True, fp16=True)
s2t_fp16.reset()

for i in range(len(speech) // chunk_size + 1):
    chunk = speech[i*chunk_size:min((i+1)*chunk_size, len(speech))]
    if len(chunk) == 0:
        break
    result = s2t_fp16(chunk, is_final=(i == len(speech) // chunk_size))
    if result and result[0][0]:
        print(f"Chunk {i+1}: '{result[0][0]}'")
        if i == 0 and result[0][2]:  # Show token IDs for first chunk
            print(f"  Token IDs: {result[0][2][:20]}")
