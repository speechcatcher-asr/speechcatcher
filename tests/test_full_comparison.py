#!/usr/bin/env python3
"""Full transcription comparison between ESPnet and ours."""

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

# Load ESPnet
from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming as ESPnetStreaming

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

print("="*80)
print("FULL TRANSCRIPTION COMPARISON")
print("="*80)

print("\n[ESPnet]")
espnet_s2t = ESPnetStreaming(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,
    ctc_weight=0.3,
)
espnet_s2t.reset()

chunk_size = 8000
espnet_outputs = []
for chunk_idx in range(len(speech) // chunk_size + 1):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    if len(chunk) == 0:
        break
    
    is_final = (chunk_idx == len(speech) // chunk_size)
    result = espnet_s2t(chunk, is_final=is_final)
    if result:
        espnet_outputs.append((chunk_idx+1, result[0][0]))
        print(f"  Chunk {chunk_idx+1}: '{result[0][0]}'")

# Load ours
print("\n[Ours]")
from speechcatcher.speechcatcher import load_model, tags

our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()

our_outputs = []
for chunk_idx in range(len(speech) // chunk_size + 1):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    if len(chunk) == 0:
        break
    
    is_final = (chunk_idx == len(speech) // chunk_size)
    result = our_s2t(chunk, is_final=is_final)
    if result:
        our_outputs.append((chunk_idx+1, result[0][0]))
        print(f"  Chunk {chunk_idx+1}: '{result[0][0]}'")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nESPnet: {' | '.join([f'C{c}: {t}' for c, t in espnet_outputs])}")
print(f"Ours:   {' | '.join([f'C{c}: {t}' for c, t in our_outputs])}")

if len(espnet_outputs) > 0 and len(our_outputs) > 0:
    espnet_final = espnet_outputs[-1][1]
    our_final = our_outputs[-1][1]
    
    print(f"\nFinal outputs:")
    print(f"  ESPnet: '{espnet_final}'")
    print(f"  Ours:   '{our_final}'")
    print(f"  Match: {'✅' if espnet_final == our_final else '❌'}")

print("\n" + "="*80)
