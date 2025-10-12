#!/usr/bin/env python3
"""Test ONLY the normalization step to isolate the difference."""

import torch
import numpy as np

# Load same features from both pipelines BEFORE normalization
from speechcatcher.speechcatcher import load_model
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming as ESPnetS2T
import wave
import hashlib
import os

print("Loading audio...")
os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
with wave.open(wavfile_path, 'rb') as wf:
    buf = wf.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

waveform = torch.from_numpy(raw_audio).float()

# Load models
print("Loading models...")
our_model = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)

espnet_model = ESPnetS2T(
    asr_train_config="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml",
    asr_model_file="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth",
    device="cpu",
)

print("\nExtracting features from frontend (before normalization)...")

# Get features BEFORE normalization using ESPnet's frontend
speech_batch = waveform.unsqueeze(0)
speech_lengths = torch.tensor([waveform.shape[0]])

with torch.no_grad():
    raw_features, feat_lengths = espnet_model.asr_model.frontend(speech_batch, speech_lengths)

print(f"Raw features: {raw_features.shape}")
print(f"Raw features stats: min={raw_features.min():.4f}, max={raw_features.max():.4f}, mean={raw_features.mean():.4f}, std={raw_features.std():.4f}")

# Now apply normalization TWO ways
print("\n" + "="*80)
print("NORMALIZATION COMPARISON")
print("="*80)

# Method 1: Our way (numpy)
print("\n[1] Our normalization (numpy-based):")
raw_np = raw_features.squeeze(0).cpu().numpy()
our_normalized_np = (raw_np - our_model.mean) / our_model.std
our_normalized = torch.from_numpy(our_normalized_np).unsqueeze(0)

print(f"  Result: min={our_normalized.min():.4f}, max={our_normalized.max():.4f}, mean={our_normalized.mean():.4f}, std={our_normalized.std():.4f}")

# Method 2: ESPnet way (torch, with masking)
print("\n[2] ESPnet normalization (torch-based with mask):")
espnet_normalized, _ = espnet_model.asr_model.normalize(raw_features, feat_lengths)

print(f"  Result: min={espnet_normalized.min():.4f}, max={espnet_normalized.max():.4f}, mean={espnet_normalized.mean():.4f}, std={espnet_normalized.std():.4f}")

# Method 3: Manual torch (no mask)
print("\n[3] Manual torch normalization (no mask):")
mean_torch = torch.from_numpy(our_model.mean).to(raw_features.dtype)
std_torch = torch.from_numpy(our_model.std).to(raw_features.dtype)
manual_normalized = (raw_features - mean_torch) / std_torch

print(f"  Result: min={manual_normalized.min():.4f}, max={manual_normalized.max():.4f}, mean={manual_normalized.mean():.4f}, std={manual_normalized.std():.4f}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

diff_1_2 = (our_normalized - espnet_normalized).abs()
print(f"\nOur vs ESPnet:")
print(f"  Max diff: {diff_1_2.max():.6f}")
print(f"  Mean diff: {diff_1_2.mean():.6f}")

diff_3_2 = (manual_normalized - espnet_normalized).abs()
print(f"\nManual torch vs ESPnet:")
print(f"  Max diff: {diff_3_2.max():.6f}")
print(f"  Mean diff: {diff_3_2.mean():.6f}")

diff_1_3 = (our_normalized - manual_normalized).abs()
print(f"\nOur vs Manual torch:")
print(f"  Max diff: {diff_1_3.max():.6f}")
print(f"  Mean diff: {diff_1_3.mean():.6f}")

if torch.allclose(our_normalized, espnet_normalized, atol=1e-4):
    print("\n✅ All normalizations match!")
else:
    print("\n❌ Normalizations differ!")

    # Check what ESPnet's mask does
    print("\n" + "="*80)
    print("CHECKING MASK EFFECT")
    print("="*80)
    from espnet_model_zoo.downloader import ModelDownloader
    from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
    from espnet2.utils.types import str2bool
    from espnet_model_zoo.downloader import ModelDownloader
    import espnet2.tasks.asr
    from espnet2.asr.frontend.default import DefaultFrontend
    # from espnet2.utils.get_default_kwargs import get_default_kwargs

    # Check if there's a mask being applied
    print(f"Feature lengths: {feat_lengths}")
    print(f"Max length: {raw_features.size(1)}")
    if feat_lengths[0] < raw_features.size(1):
        print(f"⚠️  Padding detected! {raw_features.size(1) - feat_lengths[0]} frames are padded")
    else:
        print("No padding - full sequence used")
