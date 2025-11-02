#!/usr/bin/env python3
"""Trace beam search evolution block by block to see why token 738 disappears."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("BEAM SEARCH TRACE (Block-by-Block)")
print("="*80)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as f:
    raw_audio = np.frombuffer(f.readframes(-1), dtype='int16')
speech = raw_audio.astype(np.float32) / 32768.0

# Load ESPnet token list
import json
with open("/tmp/espnet_token_list.json", "r") as f:
    espnet_token_list = json.load(f)

# Load our model - disable DEBUG logging for cleaner output
print("\n[1] Loading model...")
from speechcatcher.speechcatcher import load_model, tags

s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
s2t.reset()
print("✅ Model loaded")

# Process chunk by chunk and print beam state after each
print("\n[2] Processing chunks with beam search trace...")
chunk_size = 8000

for chunk_idx in range(6):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    is_final = False

    print(f"\n{'='*80}")
    print(f"CHUNK {chunk_idx + 1}")
    print('='*80)

    results = s2t(chunk, is_final=is_final)

    # Print beam search state
    if hasattr(s2t, 'beam_state') and s2t.beam_state is not None:
        print(f"\n--- BEAM STATE ---")
        print(f"Output index: {s2t.beam_state.output_index}")
        print(f"Processed frames: {s2t.beam_state.processed_frames}")
        print(f"Number of hypotheses: {len(s2t.beam_state.hypotheses)}")
        
        if hasattr(s2t, 'beam_search'):
            print(f"Processed blocks: {s2t.beam_search.processed_block}")
            if s2t.beam_search.encoder_buffer is not None:
                print(f"Encoder buffer shape: {s2t.beam_search.encoder_buffer.shape}")
        
        print(f"\nTop 3 hypotheses:")
        for i, hyp in enumerate(s2t.beam_state.hypotheses[:3]):
            # Decode tokens
            token_ids = hyp.yseq.tolist() if torch.is_tensor(hyp.yseq) else hyp.yseq
            tokens = [espnet_token_list[tid] for tid in token_ids]
            tokens_str = " ".join(f"{tid}:{tok}" for tid, tok in zip(token_ids, tokens))

            print(f"  [{i+1}] score={hyp.score:.4f}")
            print(f"      yseq={token_ids}")
            print(f"      tokens={tokens_str}")

    # Print results
    if results and results[0]:
        text, tokens, token_ids = results[0]
        print(f"\n--- OUTPUT ---")
        print(f"Text: '{text}'")
        print(f"Token IDs: {token_ids}")
        print(f"Tokens: {tokens}")
    else:
        print(f"\n--- OUTPUT ---")
        print("(no output)")

print("\n" + "="*80)
print("TRACE COMPLETE")
print("="*80)
