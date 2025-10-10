#!/usr/bin/env python3
"""Test if encoder states are preserved between chunks."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

# Enable debug logging for speechcatcher modules
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger('speechcatcher').setLevel(logging.DEBUG)

print("="*80)
print("STATE PRESERVATION TEST")
print("="*80)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile, load_model, tags

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

speech = raw_audio.astype(np.float32) / 32768.0

# Load model
our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()

# Process chunks
chunk_size = 8000
chunks = [speech[i*chunk_size : min((i+1)*chunk_size, len(speech))] for i in range(5)]

print("\nProcessing first 5 chunks...\n")

for chunk_idx, chunk in enumerate(chunks):
    is_final = False

    print(f"CHUNK {chunk_idx+1}:")
    print(f"  Before call:")
    print(f"    beam_state: {our_s2t.beam_state}")
    if our_s2t.beam_state:
        print(f"    beam_state.encoder_states: {our_s2t.beam_state.encoder_states}")
        if our_s2t.beam_state.encoder_states:
            print(f"    encoder_states keys: {list(our_s2t.beam_state.encoder_states.keys())[:3]}")
    print(f"    beam_search.encoder_buffer: {our_s2t.beam_search.encoder_buffer.shape if our_s2t.beam_search.encoder_buffer is not None else 'None'}")

    # Process chunk through full pipeline
    with torch.no_grad():
        results = our_s2t(chunk, is_final=is_final)

    print(f"  After call:")
    print(f"    beam_state: {our_s2t.beam_state}")
    if our_s2t.beam_state:
        print(f"    beam_state.encoder_states: {our_s2t.beam_state.encoder_states}")
        if our_s2t.beam_state.encoder_states:
            print(f"    encoder_states keys: {list(our_s2t.beam_state.encoder_states.keys())[:3]}")
    print(f"    beam_search.encoder_buffer: {our_s2t.beam_search.encoder_buffer.shape if our_s2t.beam_search.encoder_buffer is not None else 'None'}")

    if results and len(results) > 0:
        text, _, _ = results[0]
        print(f"  Result: '{text}'")
    else:
        print(f"  Result: (empty)")
    print()

print("="*80)
print("STATE PRESERVATION TEST COMPLETE")
print("="*80)
