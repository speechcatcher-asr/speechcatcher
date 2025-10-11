#!/usr/bin/env python3
"""Debug final chunk processing."""

import torch
import numpy as np
import wave
import hashlib
import os

from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as f:
    raw_audio = np.frombuffer(f.readframes(-1), dtype='int16')
speech = raw_audio.astype(np.float32) / 32768.0

# Load our model
from speechcatcher.speechcatcher import load_model, tags
import json

with open("/tmp/espnet_token_list.json", "r") as f:
    token_list = json.load(f)

our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()

chunk_size = 8000

print("="*80)
print("PROCESSING CHUNKS")
print("="*80)

# Process all chunks except the last
for chunk_idx in range(len(speech) // chunk_size):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    result = our_s2t(chunk, is_final=False)
    if result:
        print(f"Chunk {chunk_idx+1}: '{result[0][0]}'")

# Now process final chunk
final_chunk_idx = len(speech) // chunk_size
chunk = speech[final_chunk_idx*chunk_size:]
print(f"\n{'='*80}")
print(f"FINAL CHUNK {final_chunk_idx+1} (is_final=True)")
print(f"{'='*80}")
print(f"Chunk length: {len(chunk)} samples")

# Check beam state before final
if hasattr(our_s2t, 'beam_state'):
    print(f"\nBeam state BEFORE final chunk:")
    print(f"  Output index: {our_s2t.beam_state.output_index}")
    print(f"  Hypotheses: {len(our_s2t.beam_state.hypotheses)}")
    for i, hyp in enumerate(our_s2t.beam_state.hypotheses[:3]):
        token_ids = hyp.yseq.tolist() if torch.is_tensor(hyp.yseq) else hyp.yseq
        tokens = [token_list[tid] for tid in token_ids]
        print(f"  [{i+1}] score={hyp.score:.4f}")
        print(f"      yseq={token_ids[:20]}")
        print(f"      tokens={' '.join(tokens[:20])}")

result = our_s2t(chunk, is_final=True)

# Check beam state after final
if hasattr(our_s2t, 'beam_state'):
    print(f"\nBeam state AFTER final chunk:")
    print(f"  Output index: {our_s2t.beam_state.output_index}")
    print(f"  Hypotheses: {len(our_s2t.beam_state.hypotheses)}")
    for i, hyp in enumerate(our_s2t.beam_state.hypotheses[:3]):
        token_ids = hyp.yseq.tolist() if torch.is_tensor(hyp.yseq) else hyp.yseq
        tokens = [token_list[tid] for tid in token_ids]
        print(f"  [{i+1}] score={hyp.score:.4f}")
        print(f"      yseq={token_ids[:20]}")
        print(f"      tokens={' '.join(tokens[:20])}")

if result:
    print(f"\nFinal output: '{result[0][0]}'")
    print(f"Token IDs: {result[0][2][:30]}")
    print(f"Tokens: {result[0][1][:30]}")

print("\n" + "="*80)
