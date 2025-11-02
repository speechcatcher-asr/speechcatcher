#!/usr/bin/env python3
"""Check when ESPnet starts decoding."""

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

print("="*80)
print("ESPnet Decoding Start Timing")
print("="*80)

chunk_size = 8000
for chunk_idx in range(6):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    
    results = espnet_s2t(chunk, is_final=False)
    
    # Check beam state
    if hasattr(espnet_s2t, 'beam_search'):
        bs = espnet_s2t.beam_search
        if hasattr(bs, 'encbuffer') and bs.encbuffer is not None:
            print(f"\nChunk {chunk_idx+1}: encbuffer shape = {bs.encbuffer.shape}")
        else:
            print(f"\nChunk {chunk_idx+1}: encbuffer = None")
        
        if hasattr(bs, 'running') and bs.running:
            print(f"  Running hypotheses: {len(bs.running)}")
            print(f"  Top hypothesis: {bs.running[0].yseq[:5] if len(bs.running[0].yseq) < 5 else bs.running[0].yseq[:5]}")
        
    if results:
        print(f"  OUTPUT: '{results[0][0]}'")
    else:
        print(f"  OUTPUT: (none)")

print("\n" + "="*80)
