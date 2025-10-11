#!/usr/bin/env python3
"""Test FP16 on CUDA."""

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

print("Loading model with FP16 on CUDA...")
s2t = load_model(tags['de_streaming_transformer_xl'], device='cuda', beam_size=5, quiet=False, fp16=True)

print("\nModel loaded successfully!")
print(f"Model device: {next(s2t.model.parameters()).device}")
print(f"Model dtype: {next(s2t.model.parameters()).dtype}")

print("\nProcessing first chunk...")
chunk_size = 8000
chunk = speech[0:chunk_size]

try:
    result = s2t(chunk, is_final=False)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
