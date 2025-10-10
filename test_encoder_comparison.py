#!/usr/bin/env python3
"""Compare encoder outputs between our implementation and ESPnet."""

import logging
import torch
import numpy as np
import wave
import hashlib
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("ENCODER COMPUTATION COMPARISON")
logger.info("="*80)

# ============================================================================
# Load audio
# ============================================================================
logger.info("\n[1] Loading audio...")

from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

waveform = torch.from_numpy(raw_audio).float()
logger.info(f"Audio: {waveform.shape}")

# ============================================================================
# Load our model
# ============================================================================
logger.info("\n[2] Loading OUR model...")
from speechcatcher.speechcatcher import load_model

our_model = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)
logger.info("✅ Our model loaded")

# ============================================================================
# Load ESPnet model
# ============================================================================
logger.info("\n[3] Loading ESPnet model...")

from espnet2.bin.asr_inference_streaming import Speech2TextStreaming as ESPnetS2T

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_model = ESPnetS2T(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
)
logger.info("✅ ESPnet model loaded")

# ============================================================================
# Process with our frontend
# ============================================================================
logger.info("\n[4] Processing with OUR frontend...")

with torch.no_grad():
    our_features, our_lengths, _ = our_model.apply_frontend(
        speech=waveform,
        prev_states=None,
        is_final=True
    )

logger.info(f"Our features: {our_features.shape}")
logger.info(f"Our features stats: min={our_features.min():.4f}, max={our_features.max():.4f}, mean={our_features.mean():.4f}, std={our_features.std():.4f}")

# ============================================================================
# Process with ESPnet frontend
# ============================================================================
logger.info("\n[5] Processing with ESPnet frontend...")

with torch.no_grad():
    speech_batch = waveform.unsqueeze(0)
    speech_lengths = torch.tensor([waveform.shape[0]])

    espnet_features, espnet_lengths = espnet_model.asr_model.frontend(speech_batch, speech_lengths)
    espnet_features, espnet_lengths = espnet_model.asr_model.normalize(espnet_features, espnet_lengths)

logger.info(f"ESPnet features: {espnet_features.shape}")
logger.info(f"ESPnet features stats: min={espnet_features.min():.4f}, max={espnet_features.max():.4f}, mean={espnet_features.mean():.4f}, std={espnet_features.std():.4f}")

# Compare features
features_diff = (our_features - espnet_features).abs()
logger.info(f"\nFeatures difference:")
logger.info(f"  Max: {features_diff.max():.6f}")
logger.info(f"  Mean: {features_diff.mean():.6f}")

if torch.allclose(our_features, espnet_features, atol=1e-4):
    logger.info("  ✅ Features MATCH (within tolerance)")
else:
    logger.warning("  ⚠️  Features DIFFER")

# ============================================================================
# Test OUR encoder
# ============================================================================
logger.info("\n[6] Testing OUR encoder...")

with torch.no_grad():
    our_encoder_out, our_encoder_lens, our_encoder_states = our_model.model.encoder(
        our_features,
        our_lengths,
        prev_states=None,
        is_final=True,
        infer_mode=True
    )

logger.info(f"Our encoder output: {our_encoder_out.shape}")
logger.info(f"Our encoder stats: min={our_encoder_out.min():.4f}, max={our_encoder_out.max():.4f}, mean={our_encoder_out.mean():.4f}, std={our_encoder_out.std():.4f}")

# ============================================================================
# Test ESPnet encoder
# ============================================================================
logger.info("\n[7] Testing ESPnet encoder...")

with torch.no_grad():
    espnet_encoder_out, espnet_encoder_lens, espnet_encoder_states = espnet_model.asr_model.encoder(
        espnet_features,
        espnet_lengths,
        prev_states=None,
        is_final=True,
        infer_mode=True
    )

logger.info(f"ESPnet encoder output: {espnet_encoder_out.shape}")
logger.info(f"ESPnet encoder stats: min={espnet_encoder_out.min():.4f}, max={espnet_encoder_out.max():.4f}, mean={espnet_encoder_out.mean():.4f}, std={espnet_encoder_out.std():.4f}")

# ============================================================================
# Compare encoder outputs
# ============================================================================
logger.info("\n[8] Comparing encoder outputs...")

encoder_diff = (our_encoder_out - espnet_encoder_out).abs()
logger.info(f"\nEncoder output difference:")
logger.info(f"  Max: {encoder_diff.max():.6f}")
logger.info(f"  Mean: {encoder_diff.mean():.6f}")
logger.info(f"  Std: {encoder_diff.std():.6f}")

if torch.allclose(our_encoder_out, espnet_encoder_out, atol=1e-4):
    logger.info("  ✅ Encoder outputs IDENTICAL (within tolerance)")
else:
    logger.error("  ❌ Encoder outputs DIFFER!")

    # Find where biggest differences are
    max_diff_idx = encoder_diff.argmax()
    max_diff_pos = torch.unravel_index(max_diff_idx, encoder_diff.shape)
    logger.info(f"\n  Biggest diff at position {max_diff_pos}:")
    logger.info(f"    Our value: {our_encoder_out[max_diff_pos]:.6f}")
    logger.info(f"    ESPnet value: {espnet_encoder_out[max_diff_pos]:.6f}")
    logger.info(f"    Difference: {encoder_diff[max_diff_pos]:.6f}")

logger.info("\n" + "="*80)
logger.info("ENCODER COMPARISON COMPLETE")
logger.info("="*80)
