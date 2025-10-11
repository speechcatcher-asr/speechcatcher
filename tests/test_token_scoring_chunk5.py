#!/usr/bin/env python3
"""Compare scores at chunk 5 when ESPnet has actually decoded."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("TOKEN SCORING AT CHUNK 5")
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

# Load ESPnet
print("\n[1] Loading ESPnet...")
from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming as ESPnetStreaming

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_s2t = ESPnetStreaming(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,
    ctc_weight=0.3,
)
espnet_s2t.reset()
print("✅ ESPnet loaded")

# Load ours
print("\n[2] Loading our implementation...")
from speechcatcher.speechcatcher import load_model, tags

our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()
print("✅ Ours loaded")

# Process chunks 1-5
print("\n[3] Processing chunks 1-5...")
chunk_size = 8000

for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    
    # ESPnet
    espnet_results = espnet_s2t(chunk, is_final=False)
    if espnet_results:
        print(f"  Chunk {chunk_idx+1} - ESPnet output: {espnet_results[0]}")
    
    # Ours
    our_results = our_s2t(chunk, is_final=False)
    if our_results:
        print(f"  Chunk {chunk_idx+1} - Our output: {our_results[0]}")

print("✅ Processed chunks 1-5")

# Compare beam states
print("\n" + "="*80)
print("BEAM STATE AFTER CHUNK 5")
print("="*80)

# ESPnet beam state
print("\n[ESPnet] Top 5 hypotheses:")
if hasattr(espnet_s2t, 'beam_state') and espnet_s2t.beam_state is not None and espnet_s2t.beam_state.hypotheses:
    for i, hyp in enumerate(espnet_s2t.beam_state.hypotheses[:5]):
        token_ids = hyp.yseq.tolist() if torch.is_tensor(hyp.yseq) else hyp.yseq
        tokens = [espnet_token_list[tid] for tid in token_ids]
        print(f"  [{i+1}] score={hyp.score:.6f}")
        print(f"      yseq={token_ids}")
        print(f"      tokens={' '.join(tokens)}")
else:
    print("  (no hypotheses)")

# Our beam state
print("\n[Ours] Top 5 hypotheses:")
if hasattr(our_s2t, 'beam_state') and our_s2t.beam_state is not None and our_s2t.beam_state.hypotheses:
    for i, hyp in enumerate(our_s2t.beam_state.hypotheses[:5]):
        token_ids = hyp.yseq.tolist() if torch.is_tensor(hyp.yseq) else hyp.yseq
        tokens = [espnet_token_list[tid] for tid in token_ids]
        print(f"  [{i+1}] score={hyp.score:.6f}")
        print(f"      yseq={token_ids}")
        print(f"      tokens={' '.join(tokens)}")
else:
    print("  (no hypotheses)")

print("\n" + "="*80)
