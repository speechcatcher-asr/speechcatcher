#!/usr/bin/env python3
"""Check BBD state to see if rollback is happening correctly."""

import numpy as np
import wave
import hashlib
import os
import torch

print("="*80)
print("BBD STATE TRACE")
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

# Load our implementation
print("\nLoading model...")
from speechcatcher.speechcatcher import load_model, tags

s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
s2t.reset()
print("✅ Model loaded")

# Load BPE for decoding
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/data/de_token_list/bpe_unigram1024/bpe.model")

# Process chunks
chunk_size = 8000
print("\nProcessing chunks with BBD trace...")

for chunk_idx in range(6):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    is_final = False

    print(f"\n{'='*80}")
    print(f"CHUNK {chunk_idx+1}")
    print(f"{'='*80}")

    # Check state before
    if hasattr(s2t, 'beam_search') and hasattr(s2t.beam_search, 'state'):
        bs = s2t.beam_search
        if bs.state and bs.state.hypotheses:
            print(f"\nBefore processing:")
            print(f"  output_index: {bs.state.output_index}")
            print(f"  processed_frames: {bs.state.processed_frames}")
            print(f"  Top hypothesis:")
            hyp = bs.state.hypotheses[0]
            yseq = hyp.yseq.tolist()
            tokens = [sp.IdToPiece(t) for t in yseq]
            print(f"    yseq: {yseq}")
            print(f"    tokens: {tokens}")
            print(f"    score: {hyp.score:.4f}")

    # Process
    results = s2t(chunk, is_final=is_final)

    # Check state after
    if hasattr(s2t, 'beam_search') and hasattr(s2t.beam_search, 'state'):
        bs = s2t.beam_search
        if bs.state and bs.state.hypotheses:
            print(f"\nAfter processing:")
            print(f"  output_index: {bs.state.output_index}")
            print(f"  processed_frames: {bs.state.processed_frames}")
            print(f"  Top hypothesis:")
            hyp = bs.state.hypotheses[0]
            yseq = hyp.yseq.tolist()
            tokens = [sp.IdToPiece(t) for t in yseq]
            print(f"    yseq: {yseq}")
            print(f"    tokens: {tokens}")
            print(f"    score: {hyp.score:.4f}")

    # Show results
    print(f"\nResults:")
    if results and results[0][0]:
        result = results[0]
        print(f"  Text: \"{result[0]}\"")
        print(f"  Tokens: {result[1]}")
        print(f"  Token IDs: {result[2]}")

        # Extract just the NEW tokens output this chunk
        if chunk_idx > 0 and prev_output_index is not None:
            new_token_count = bs.state.output_index - prev_output_index
            print(f"  New tokens this chunk: {new_token_count}")
    else:
        print(f"  (no output)")

    # Save output_index for next iteration
    if hasattr(s2t, 'beam_search') and hasattr(s2t.beam_search, 'state'):
        prev_output_index = s2t.beam_search.state.output_index
    else:
        prev_output_index = None

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
