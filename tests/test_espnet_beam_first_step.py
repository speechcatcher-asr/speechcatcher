#!/usr/bin/env python3
"""Trace ESPnet's beam search at first decoding step."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')

from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as f:
    raw_audio = np.frombuffer(f.readframes(-1), dtype='int16')
speech = raw_audio.astype(np.float32) / 32768.0

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

chunk_size = 8000

# Process chunks 1-4 (no decoding yet)
for chunk_idx in range(4):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    espnet_s2t(chunk, is_final=False)

print("="*80)
print("Before chunk 5 (no decoding yet)")
print("="*80)
print(f"encbuffer: {espnet_s2t.beam_search.encbuffer.shape if espnet_s2t.beam_search.encbuffer is not None else None}")
print(f"running_hyps: {len(espnet_s2t.beam_search.running_hyps) if espnet_s2t.beam_search.running_hyps else 0}")

# Process chunk 5 (first decoding)
print("\n" + "="*80)
print("Processing chunk 5 (with DEBUG logging)...")
print("="*80)

chunk = speech[4*chunk_size : min(5*chunk_size, len(speech))]
result = espnet_s2t(chunk, is_final=False)

print("\n" + "="*80)
print("After chunk 5")
print("="*80)
print(f"Output: {result}")

if espnet_s2t.beam_search.running_hyps:
    print(f"\nTop 5 hypotheses:")
    for i, hyp in enumerate(espnet_s2t.beam_search.running_hyps[:5]):
        print(f"  [{i+1}] score={hyp.score:.6f}, yseq={hyp.yseq.tolist()}")
