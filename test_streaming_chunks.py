#!/usr/bin/env python3
"""Test BBD with multiple streaming chunks."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

# Enable debug logging for beam search
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('speechcatcher.beam_search.beam_search').setLevel(logging.DEBUG)

print("="*80)
print("STREAMING CHUNKS TEST (BBD)")
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

# Reset state
speech2text.reset()

# Process in CHUNKS (streaming mode)
print("\n[3] Processing in streaming chunks...")

chunk_size = 8000  # ~0.5s at 16kHz
num_chunks = (len(speech) + chunk_size - 1) // chunk_size

print(f"Total chunks: {num_chunks}")

results_list = []
for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(speech))
    chunk = speech[start:end]
    is_final = (i == num_chunks - 1)

    print(f"\n--- Chunk {i+1}/{num_chunks} (is_final={is_final}) ---")
    results = speech2text(speech=chunk, is_final=is_final)

    if results and len(results) > 0:
        text, tokens, token_ids = results[0]
        print(f"Result: '{text[:50]}...' ({len(tokens)} tokens)")
        results_list.append((text, tokens, token_ids))

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if results_list:
    final_text, final_tokens, final_token_ids = results_list[-1]
    print(f"\n✅ Final text: '{final_text}'")
    print(f"\n✅ Token count: {len(final_tokens)}")

    # Check for Arabic token
    if 'م' in final_tokens:
        count = final_tokens.count('م')
        print(f"\n⚠️  Arabic 'م' appears {count} times")
    else:
        print(f"\n✅ No Arabic characters!")
else:
    print("\n❌ No results!")

print("\n" + "="*80)
print("STREAMING CHUNKS TEST COMPLETE")
print("="*80)
