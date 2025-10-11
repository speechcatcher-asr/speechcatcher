#!/usr/bin/env python3
"""Trace ESPnet's beam search to see what it does with token 1023."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("ESPNET BEAM SEARCH TRACE (Why does it output 'liebe'?)")
print("="*80)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

speech = raw_audio.astype(np.float32) / 32768.0

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

# Process chunks 1-5
print("\n[2] Processing chunks 1-5...")
chunk_size = 8000

for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    is_final = False

    results = espnet_s2t(chunk, is_final=is_final)

    if chunk_idx == 4:
        print(f"\nChunk {chunk_idx+1} results:")
        if results and len(results) > 0:
            print(f"  Results: {results}")
            # ESPnet returns different format
            result = results[0]
            print(f"  Result type: {type(result)}")
            print(f"  Result: {result}")
        else:
            print(f"  (no results yet)")

        # Check beam search state
        print(f"\nBeam search state:")
        print(f"  processed_block: {espnet_s2t.beam_search.processed_block}")
        if hasattr(espnet_s2t.beam_search, 'encbuffer'):
            print(f"  encbuffer shape: {espnet_s2t.beam_search.encbuffer.shape if espnet_s2t.beam_search.encbuffer is not None else 'None'}")
        if hasattr(espnet_s2t.beam_search, 'running_hyps'):
            print(f"  running_hyps: {len(espnet_s2t.beam_search.running_hyps)} hypotheses")
            # Show hypotheses
            for i, hyp in enumerate(espnet_s2t.beam_search.running_hyps[:3]):
                print(f"    Hyp {i}: yseq={hyp.yseq.tolist()[:15]}, score={hyp.score:.4f}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
