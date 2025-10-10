#!/usr/bin/env python3
"""Test full transcription with debug output."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

print("="*80)
print("FULL TRANSCRIPTION TEST")
print("="*80)

# Load audio
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
print("\nLoading model...")
our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()
print("✅ Loaded\n")

# Process chunks
chunk_size = 8000
num_chunks = (len(speech) + chunk_size - 1) // chunk_size

print(f"Processing {num_chunks} chunks of {chunk_size} samples each\n")

for chunk_idx in range(min(10, num_chunks)):
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, len(speech))
    chunk = speech[start:end]
    is_final = (chunk_idx == num_chunks - 1)

    print(f"{'='*60}")
    print(f"CHUNK {chunk_idx+1}/{num_chunks} (is_final={is_final})")
    print(f"{'='*60}")

    with torch.no_grad():
        results = our_s2t(chunk, is_final=is_final)

    # Debug hypotheses
    if our_s2t.beam_state and our_s2t.beam_state.hypotheses:
        best_hyp = our_s2t.beam_state.hypotheses[0]
        print(f"Best hypothesis:")
        print(f"  yseq: {best_hyp.yseq.tolist()}")
        print(f"  score: {best_hyp.score:.4f}")

    # Check buffer
    if our_s2t.beam_search.encoder_buffer is not None:
        print(f"Encoder buffer: {our_s2t.beam_search.encoder_buffer.shape}")
        print(f"Processed blocks: {our_s2t.beam_search.processed_block}")

    # Results
    if results and len(results) > 0:
        text, tokens, token_ids = results[0]
        print(f"Result text: '{text}'")
        print(f"Token IDs: {token_ids}")
    else:
        print("Result: (empty)")

    print()

print("="*80)
print("TEST COMPLETE")
print("="*80)
