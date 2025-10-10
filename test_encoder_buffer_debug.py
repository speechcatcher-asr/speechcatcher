#!/usr/bin/env python3
"""Debug encoder buffer accumulation."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

# Enable ALL debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

print("="*80)
print("ENCODER BUFFER DEBUG")
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
    raw_audio = np.frombuffer(buf, dtype='int16')

speech = raw_audio.astype(np.float32) / 32768.0
print(f"Audio: {len(speech)} samples ({len(speech)/rate:.2f}s)")

# Load model
print("\n[2] Loading model...")
our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()
print(f"✅ Loaded")

# Process in chunks with detailed logging
print("\n" + "="*80)
print("PROCESSING CHUNKS WITH DEBUG")
print("="*80)

chunk_size = 8000
chunks = [
    speech[i*chunk_size : min((i+1)*chunk_size, len(speech))]
    for i in range((len(speech) + chunk_size - 1) // chunk_size)
]

print(f"\nTotal chunks: {len(chunks)}")
print(f"Chunk sizes: {[len(c) for c in chunks]}\n")

for chunk_idx, chunk in enumerate(chunks[:5]):  # Process first 5 chunks
    is_final = (chunk_idx == len(chunks) - 1)

    print(f"\n{'='*80}")
    print(f"CHUNK {chunk_idx+1}/{len(chunks)} (is_final={is_final})")
    print(f"{'='*80}")

    print(f"\n[Before] Encoder buffer: {our_s2t.beam_search.encoder_buffer.shape if our_s2t.beam_search.encoder_buffer is not None else 'None'}")
    print(f"[Before] Processed blocks: {our_s2t.beam_search.processed_block}")

    # Process chunk
    with torch.no_grad():
        results = our_s2t(chunk, is_final=is_final)

    print(f"\n[After] Encoder buffer: {our_s2t.beam_search.encoder_buffer.shape if our_s2t.beam_search.encoder_buffer is not None else 'None'}")
    print(f"[After] Processed blocks: {our_s2t.beam_search.processed_block}")

    if results and len(results) > 0:
        text, tokens, token_ids = results[0]
        print(f"\n✅ Result: '{text}' ({len(tokens)} tokens)")
    else:
        print(f"\n⚠️  No results yet")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
