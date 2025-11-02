#!/usr/bin/env python3
"""Trace beam state after each chunk for both implementations."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

from speechcatcher.speechcatcher import convert_inputfile, load_model, tags

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as f:
    raw_audio = np.frombuffer(f.readframes(-1), dtype='int16')
speech = raw_audio.astype(np.float32) / 32768.0

# Load our model
our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()

chunk_size = 8000

print("="*80)
print("OUR IMPLEMENTATION")
print("="*80)

for chunk_idx in range(len(speech) // chunk_size + 1):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    if len(chunk) == 0:
        break
    result = our_s2t(chunk, is_final=False)

    # Show beam state
    if hasattr(our_s2t, 'beam_state'):
        print(f"\nAfter Chunk {chunk_idx+1}:")
        if result:
            print(f"  Output: '{result[0][0]}'")
        else:
            print(f"  Output: (none)")
        print(f"  Output index: {our_s2t.beam_state.output_index}")
        print(f"  Beam size: {len(our_s2t.beam_state.hypotheses)}")
        for i, hyp in enumerate(our_s2t.beam_state.hypotheses[:3]):
            yseq = hyp.yseq.tolist()[:10]
            has_eos = hyp.yseq[-1].item() == 1023
            yseq_len = len(hyp.yseq) - 1  # Excluding SOS
            print(f"    [{i+1}] len={yseq_len}, yseq={yseq}, score={hyp.score:.4f}, has_eos={has_eos}")
