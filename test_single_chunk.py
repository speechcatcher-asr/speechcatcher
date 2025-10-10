#!/usr/bin/env python3
"""Test with a single large chunk to isolate streaming issues."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("SINGLE CHUNK TEST")
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
print("\n[2] Loading model...")
speech2text = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
print("✅ Model loaded")

# Process as a SINGLE chunk with is_final=True
print("\n[3] Processing as single chunk...")
results = speech2text(speech=speech, is_final=True)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if results and len(results) > 0:
    text, tokens, token_ids = results[0]
    print(f"\n✅ Text: '{text}'")
    print(f"\n✅ Tokens ({len(tokens)}): {tokens[:20]}...")
    print(f"\n✅ Token IDs ({len(token_ids)}): {token_ids[:20]}...")

    # Check for Arabic token
    if 'م' in tokens:
        count = tokens.count('م')
        print(f"\n⚠️  Arabic 'م' appears {count} times")
    elif 1023 in token_ids:
        count = token_ids.count(1023)
        print(f"\n⚠️  Token ID 1023 appears {count} times")
    else:
        print(f"\n✅ No problematic tokens!")
else:
    print("\n❌ No results!")

print("\n" + "="*80)
print("SINGLE CHUNK TEST COMPLETE")
print("="*80)
