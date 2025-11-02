#!/usr/bin/env python3
"""Test normalization step."""

import logging
import sys
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile
import wave
import hashlib
import os

logger.info("Loading audio...")
os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

waveform = torch.from_numpy(raw_audio).float()

# Extract features
from speechcatcher.model.frontend.stft_frontend import STFTFrontend

frontend = STFTFrontend(n_fft=512, hop_length=160, win_length=400, n_mels=80)
with torch.no_grad():
    features, lengths = frontend(waveform)

logger.info(f"Features (before norm): shape={features.shape}")
logger.info(f"Features stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}, std={features.std():.4f}")

# Load model to get normalization stats
from speechcatcher.speechcatcher import load_model

speech2text = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)

# Check if model has normalization
if hasattr(speech2text.model, 'normalize'):
    logger.info(f"\n✅ Model has normalize module: {speech2text.model.normalize}")
    logger.info(f"Normalize type: {type(speech2text.model.normalize)}")

    # Check normalization stats
    if hasattr(speech2text.model.normalize, 'mean'):
        logger.info(f"Norm mean shape: {speech2text.model.normalize.mean.shape}")
        logger.info(f"Norm mean stats: min={speech2text.model.normalize.mean.min():.4f}, max={speech2text.model.normalize.mean.max():.4f}")
    if hasattr(speech2text.model.normalize, 'std'):
        logger.info(f"Norm std shape: {speech2text.model.normalize.std.shape}")
        logger.info(f"Norm std stats: min={speech2text.model.normalize.std.min():.4f}, max={speech2text.model.normalize.std.max():.4f}")

    # Apply normalization manually
    logger.info("\n" + "="*60)
    logger.info("Applying normalization...")
    logger.info("="*60)

    with torch.no_grad():
        normalized_features = speech2text.model.normalize(features)

    logger.info(f"Normalized features: shape={normalized_features.shape}")
    logger.info(f"Normalized stats: min={normalized_features.min():.4f}, max={normalized_features.max():.4f}, mean={normalized_features.mean():.4f}, std={normalized_features.std():.4f}")

else:
    logger.warning("❌ Model does NOT have normalize module!")

# Check if normalization is in the model's forward pass
logger.info("\n" + "="*60)
logger.info("Testing full model forward pass...")
logger.info("="*60)

# Run through the model's frontend + normalize
with torch.no_grad():
    # Frontend
    model_features, model_lengths = speech2text.model.frontend(waveform.unsqueeze(0))
    logger.info(f"Model frontend output: shape={model_features.shape}")
    logger.info(f"Model frontend stats: min={model_features.min():.4f}, max={model_features.max():.4f}, mean={model_features.mean():.4f}")

    # Normalize (if exists)
    if hasattr(speech2text.model, 'normalize'):
        model_features = speech2text.model.normalize(model_features)
        logger.info(f"After normalize: shape={model_features.shape}")
        logger.info(f"After normalize stats: min={model_features.min():.4f}, max={model_features.max():.4f}, mean={model_features.mean():.4f}, std={model_features.std():.4f}")

    # Encoder
    logger.info("\nRunning encoder...")
    encoder_out, encoder_out_lens, encoder_states = speech2text.model.encoder(
        model_features,
        model_lengths,
        prev_states=None,
        is_final=True,
        infer_mode=True
    )

    logger.info(f"Encoder output: shape={encoder_out.shape}")
    logger.info(f"Encoder stats: min={encoder_out.min():.4f}, max={encoder_out.max():.4f}, mean={encoder_out.mean():.4f}, std={encoder_out.std():.4f}")

    # Check for NaN/Inf
    if torch.isnan(encoder_out).any():
        logger.error("❌ Encoder output contains NaN!")
    if torch.isinf(encoder_out).any():
        logger.error("❌ Encoder output contains Inf!")
    if not torch.isnan(encoder_out).any() and not torch.isinf(encoder_out).any():
        logger.info("✅ Encoder output is clean (no NaN/Inf)")

logger.info("\n" + "="*60)
logger.info("Normalization test complete!")
logger.info("="*60)
