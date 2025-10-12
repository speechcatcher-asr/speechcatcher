#!/usr/bin/env python3
"""Debug BBD behavior."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

print("="*80)
print("BBD DEBUG TEST")
print("="*80)

# Load audio
print("\n[1] Loading audio...")
from speechcatcher.speechcatcher import convert_inputfile, load_model, tags

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    rate = wavfile_in.getframerate()
    buf = wavfile_in.readframes(-1)
    speech = np.frombuffer(buf, dtype='int16')

# Normalize to [-1, 1]
speech = speech.astype(np.float32) / 32768.0

print(f"Audio: rate={rate} Hz, shape={speech.shape}, range=[{speech.min():.4f}, {speech.max():.4f}]")

# Load model
print("\n[2] Loading model with BBD enabled...")
speech2text = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
print("✅ Model loaded")
print(f"BBD enabled: {speech2text.beam_search.use_bbd}")
print(f"BBD conservative: {speech2text.beam_search.bbd_conservative}")

# Process as a SINGLE chunk with is_final=True
print("\n[3] Processing as single chunk...")
results = speech2text(speech=speech, is_final=True)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if results and len(results) > 0:
    text, tokens, token_ids = results[0]
    print(f"\n✅ Text: '{text}'")
    print(f"\n✅ Token count: {len(tokens)}")

    # Check for Arabic token
    if 'م' in tokens:
        count = tokens.count('م')
        print(f"\n⚠️  Arabic 'م' appears {count} times")
    else:
        print(f"\n✅ No Arabic characters!")
else:
    print("\n❌ No results!")

print("\n" + "="*80)
print("BBD DEBUG TEST COMPLETE")
print("="*80)
