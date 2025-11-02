#!/usr/bin/env python3
"""Test batch (non-streaming) mode."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("BATCH MODE TEST")
print("="*80)

# Load audio
print("\n[1] Loading audio...")
from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    rate = wavfile_in.getframerate()
    buf = wavfile_in.readframes(-1)
    speech = np.frombuffer(buf, dtype='int16')

print(f"Audio: rate={rate} Hz, shape={speech.shape}")

# Load ESPnet model in BATCH mode (no streaming)
print("\n[2] Loading ESPnet model in BATCH mode...")
from espnet2.bin.asr_inference import Speech2Text

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_batch = Speech2Text(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,
)
print("✅ ESPnet batch model loaded")

# Transcribe with batch mode
print("\n[3] Running batch transcription...")

# ESPnet expects float audio normalized to [-1, 1]
speech_float = speech.astype(np.float32) / 32768.0
print(f"Audio range: [{speech_float.min():.4f}, {speech_float.max():.4f}]")

nbests = espnet_batch(speech_float)
text = nbests[0][0]  # Get best hypothesis text

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\n✅ Transcription: '{text}'")

print("\n" + "="*80)
print("BATCH MODE TEST COMPLETE")
print("="*80)
