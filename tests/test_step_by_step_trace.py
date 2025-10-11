#!/usr/bin/env python3
"""Step-by-step trace to see where beam searches diverge."""

import numpy as np
import wave
import hashlib
import os
import torch

print("="*80)
print("STEP-BY-STEP BEAM SEARCH TRACE")
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

# Load BPE for decoding
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/data/de_token_list/bpe_unigram1024/bpe.model")

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

    espnet_results = espnet_s2t(chunk, is_final=False)
    our_results = our_s2t(chunk, is_final=False)

print("\n" + "="*80)
print("CHUNK 5 - HYPOTHESIS COMPARISON")
print("="*80)

# ESPnet hypotheses
print("\n[ESPnet]")
if hasattr(espnet_s2t.beam_search, 'running_hyps') and espnet_s2t.beam_search.running_hyps:
    print(f"Number of hypotheses: {len(espnet_s2t.beam_search.running_hyps)}")
    for i, hyp in enumerate(espnet_s2t.beam_search.running_hyps[:3]):
        if hasattr(hyp, 'yseq'):
            yseq = hyp.yseq.tolist() if torch.is_tensor(hyp.yseq) else hyp.yseq
            tokens = [sp.IdToPiece(t) for t in yseq]
            score = hyp.score.item() if torch.is_tensor(hyp.score) else hyp.score
            print(f"  Hyp {i+1}:")
            print(f"    yseq: {yseq}")
            print(f"    tokens: {tokens}")
            print(f"    score: {score:.4f}")

# Our hypotheses
print("\n[Ours]")
if hasattr(our_s2t, 'beam_state') and our_s2t.beam_state and our_s2t.beam_state.hypotheses:
    print(f"Number of hypotheses: {len(our_s2t.beam_state.hypotheses)}")
    print(f"output_index: {our_s2t.beam_state.output_index}")
    for i, hyp in enumerate(our_s2t.beam_state.hypotheses[:3]):
        yseq = hyp.yseq.tolist()
        tokens = [sp.IdToPiece(t) for t in yseq]
        print(f"  Hyp {i+1}:")
        print(f"    yseq: {yseq}")
        print(f"    tokens: {tokens}")
        print(f"    score: {hyp.score:.4f}")

# Compare outputs
print("\n" + "="*80)
print("OUTPUT COMPARISON")
print("="*80)

espnet_text = espnet_results[0][0] if espnet_results and espnet_results[0][0] else "(none)"
our_text = our_results[0][0] if our_results and our_results[0][0] else "(none)"

print(f"\nESPnet: \"{espnet_text}\"")
print(f"Ours:   \"{our_text}\"")

if espnet_text == our_text:
    print("\n✅ OUTPUTS MATCH!")
else:
    print(f"\n❌ OUTPUTS DIFFER!")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
