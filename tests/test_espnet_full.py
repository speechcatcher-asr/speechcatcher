#!/usr/bin/env python3
"""Test ESPnet's full transcription pipeline."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("ESPNET FULL PIPELINE TEST")
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

# Load ESPnet model
print("\n[2] Loading ESPnet model...")
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming as ESPnetS2T

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_model = ESPnetS2T(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,  # Match our beam size
)
print("✅ ESPnet model loaded")

# Transcribe
print("\n[3] Running ESPnet transcription...")

# ESPnet expects float audio normalized to [-1, 1]
speech_float = speech.astype(np.float32)

# Try different chunk sizes to match our streaming
chunk_size = 8000  # About 0.5s at 16kHz

results = []
for i in range(0, len(speech_float), chunk_size):
    chunk = speech_float[i:i + chunk_size]
    is_final = (i + chunk_size >= len(speech_float))

    result = espnet_model(chunk, is_final=is_final)
    if result:
        results.append(result)
        print(f"Chunk {i//chunk_size + 1}: '{result[0]}'")

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if results:
    final_text = results[-1][0] if results else ""
    print(f"\n✅ Final text: '{final_text}'")

    # Compare with expected
    expected_path = "Neujahrsansprache_5s.mp4.txt.expected"
    if os.path.exists(expected_path):
        with open(expected_path, 'r') as f:
            expected = f.read().strip()
        print(f"\n📋 Expected: '{expected}'")

        if final_text.strip() == expected:
            print("\n✅ PERFECT MATCH!")
        else:
            print("\n❌ MISMATCH!")
else:
    print("\n❌ No results produced!")

print("\n" + "="*80)
print("ESPNET FULL PIPELINE TEST COMPLETE")
print("="*80)
