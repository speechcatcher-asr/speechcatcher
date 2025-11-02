#!/usr/bin/env python3
"""Test waveform buffering - does it wait for enough samples?"""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("WAVEFORM BUFFERING TEST")
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

print(f"Frontend params: win_length={our_s2t.win_length}, hop_length={our_s2t.hop_length}")
print(f"Minimum samples needed: {our_s2t.win_length + 1}")

# Test with VERY small chunks (smaller than win_length)
print("\n" + "="*80)
print("TEST 1: Chunks smaller than win_length")
print("="*80)

our_s2t.reset()
chunk_size = 100  # Much smaller than win_length=400

print(f"\nChunk size: {chunk_size} samples (< win_length={our_s2t.win_length})")
print("Expected: Chunks 1-3 should buffer, chunk 4 should process\n")

for chunk_idx in range(6):
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, len(speech))
    chunk = speech[start:end]

    is_final = False

    print(f"Chunk {chunk_idx+1}: {len(chunk)} samples")

    # Check frontend state BEFORE
    buffered_before = 0
    if our_s2t.frontend_states and "waveform_buffer" in our_s2t.frontend_states:
        buffered_before = our_s2t.frontend_states["waveform_buffer"].size(0)

    with torch.no_grad():
        results = our_s2t(chunk, is_final=is_final)

    # Check frontend state AFTER
    buffered_after = 0
    if our_s2t.frontend_states and "waveform_buffer" in our_s2t.frontend_states:
        buffered_after = our_s2t.frontend_states["waveform_buffer"].size(0)

    # Check encoder buffer
    enc_buf_size = 0
    if our_s2t.beam_search.encoder_buffer is not None:
        enc_buf_size = our_s2t.beam_search.encoder_buffer.size(1)

    print(f"  Waveform buffer: {buffered_before} → {buffered_after} samples")
    print(f"  Encoder buffer: {enc_buf_size} frames")
    print(f"  Results: {'(empty)' if not results or len(results) == 0 or not results[0][0] else results[0][0]}")
    print()

# Test with normal chunks
print("="*80)
print("TEST 2: Normal chunks (larger than win_length)")
print("="*80)

our_s2t.reset()
chunk_size = 8000

print(f"\nChunk size: {chunk_size} samples (>> win_length={our_s2t.win_length})")
print("Expected: Every chunk should process immediately\n")

for chunk_idx in range(3):
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, len(speech))
    chunk = speech[start:end]

    is_final = False

    print(f"Chunk {chunk_idx+1}: {len(chunk)} samples")

    buffered_before = 0
    if our_s2t.frontend_states and "waveform_buffer" in our_s2t.frontend_states:
        buffered_before = our_s2t.frontend_states["waveform_buffer"].size(0)

    with torch.no_grad():
        results = our_s2t(chunk, is_final=is_final)

    buffered_after = 0
    if our_s2t.frontend_states and "waveform_buffer" in our_s2t.frontend_states:
        buffered_after = our_s2t.frontend_states["waveform_buffer"].size(0)

    enc_buf_size = 0
    if our_s2t.beam_search.encoder_buffer is not None:
        enc_buf_size = our_s2t.beam_search.encoder_buffer.size(1)

    print(f"  Waveform buffer: {buffered_before} → {buffered_after} samples")
    print(f"  Encoder buffer: {enc_buf_size} frames")
    print()

print("="*80)
print("TEST COMPLETE")
print("="*80)
print("\nConclusion:")
print("- Small chunks should accumulate in waveform buffer until > win_length")
print("- Large chunks should process immediately with STFT overlap buffering")
