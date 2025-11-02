#!/usr/bin/env python3
"""Compare frontend processing between our implementation and ESPnet."""

import logging
import sys
import numpy as np
import torch

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Load audio using speechcatcher's ffmpeg function (same way as recognize_file)
from speechcatcher.speechcatcher import convert_inputfile
import wave
import hashlib
import os

logger.info("Loading audio with speechcatcher's ffmpeg loader...")

# Use same approach as recognize_file: convert to temp WAV file
os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

# Read WAV file
with wave.open(wavfile_path, 'rb') as wavfile_in:
    ch = wavfile_in.getnchannels()
    bits = wavfile_in.getsampwidth()
    rate = wavfile_in.getframerate()
    nframes = wavfile_in.getnframes()
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

logger.info(f"Audio loaded: shape={raw_audio.shape}, rate={rate}, channels={ch}, bits={bits*8}")
logger.info(f"Audio stats: min={raw_audio.min():.4f}, max={raw_audio.max():.4f}, mean={raw_audio.mean():.4f}")

# Convert to torch tensor
waveform = torch.from_numpy(raw_audio).float()
logger.info(f"Waveform tensor: shape={waveform.shape}, dtype={waveform.dtype}")

# ============================================
# TEST 1: Our Frontend Implementation
# ============================================
logger.info("\n" + "="*60)
logger.info("TEST 1: Our STFTFrontend Implementation")
logger.info("="*60)

from speechcatcher.model.frontend.stft_frontend import STFTFrontend

# Create frontend with ESPnet default parameters
our_frontend = STFTFrontend(
    n_fft=512,
    hop_length=160,
    win_length=400,
    n_mels=80,
    sample_rate=16000,
    f_min=0.0,
    f_max=None,
)

logger.info(f"Frontend config: n_fft={our_frontend.n_fft}, hop_length={our_frontend.hop_length}, "
           f"win_length={our_frontend.win_length}, n_mels={our_frontend.n_mels}")

# Extract features
with torch.no_grad():
    our_features, our_lengths = our_frontend(waveform)

logger.info(f"Our features: shape={our_features.shape}, dtype={our_features.dtype}")
logger.info(f"Our lengths: {our_lengths}")
logger.info(f"Our feature stats: min={our_features.min():.4f}, max={our_features.max():.4f}, mean={our_features.mean():.4f}")

# Check for NaN/Inf
if torch.isnan(our_features).any():
    logger.error("Our features contain NaN!")
if torch.isinf(our_features).any():
    logger.error("Our features contain Inf!")

# ============================================
# TEST 2: Check torch.stft directly
# ============================================
logger.info("\n" + "="*60)
logger.info("TEST 2: Direct torch.stft Call")
logger.info("="*60)

# Call torch.stft directly to see raw STFT output
window = torch.hann_window(400)
stft_result = torch.stft(
    waveform,
    n_fft=512,
    hop_length=160,
    win_length=400,
    window=window,
    center=True,
    pad_mode='reflect',
    normalized=False,
    onesided=True,
    return_complex=True
)

logger.info(f"STFT result: shape={stft_result.shape}, dtype={stft_result.dtype}")
logger.info(f"STFT real min/max: {stft_result.real.min():.4f} / {stft_result.real.max():.4f}")
logger.info(f"STFT imag min/max: {stft_result.imag.min():.4f} / {stft_result.imag.max():.4f}")

# Compute power spectrum
power_spec = stft_result.abs().pow(2)
logger.info(f"Power spectrum: shape={power_spec.shape}")
logger.info(f"Power spectrum stats: min={power_spec.min():.4f}, max={power_spec.max():.4f}, mean={power_spec.mean():.4f}")

# ============================================
# TEST 3: Load actual ESPnet model and compare
# ============================================
logger.info("\n" + "="*60)
logger.info("TEST 3: ESPnet Model Frontend")
logger.info("="*60)

from speechcatcher.speechcatcher import load_model

# Load model (which includes ESPnet frontend)
speech2text = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)

# Access the frontend from the loaded model
logger.info(f"Model frontend type: {type(speech2text.model.frontend)}")
logger.info(f"Model frontend: {speech2text.model.frontend}")

# Check if model has a separate frontend attribute
if hasattr(speech2text.model, 'frontend'):
    frontend = speech2text.model.frontend
    logger.info(f"Frontend attributes: {dir(frontend)}")

    # Try to extract features using model's frontend
    try:
        with torch.no_grad():
            if hasattr(frontend, 'forward'):
                espnet_features, espnet_lengths = frontend(waveform.unsqueeze(0))
                logger.info(f"ESPnet features: shape={espnet_features.shape}")
                logger.info(f"ESPnet feature stats: min={espnet_features.min():.4f}, max={espnet_features.max():.4f}, mean={espnet_features.mean():.4f}")

                # Compare with our implementation
                logger.info("\n" + "="*60)
                logger.info("COMPARISON: Our vs ESPnet Frontend")
                logger.info("="*60)

                # Our features already have batch dim, just need to match
                logger.info(f"Shape before: ours={our_features.shape} vs espnet={espnet_features.shape}")

                if our_features.shape == espnet_features.shape:
                    diff = (our_features - espnet_features).abs()
                    logger.info(f"Absolute difference: min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}")

                    # Check relative difference
                    rel_diff = diff / (espnet_features.abs() + 1e-10)
                    logger.info(f"Relative difference: min={rel_diff.min():.6f}, max={rel_diff.max():.6f}, mean={rel_diff.mean():.6f}")

                    # Check if they're close
                    if torch.allclose(our_features, espnet_features, rtol=1e-3, atol=1e-5):
                        logger.info("✅ Features are CLOSE (within tolerance)!")
                    else:
                        logger.warning("❌ Features are DIFFERENT!")

                        # Show first few frames for debugging
                        logger.info("\nFirst frame comparison:")
                        logger.info(f"Ours:   {our_features[0, 0, :10]}")
                        logger.info(f"ESPnet: {espnet_features[0, 0, :10]}")
                else:
                    logger.warning(f"Shape mismatch! Cannot compare.")
    except Exception as e:
        logger.error(f"Error extracting ESPnet features: {e}", exc_info=True)

# ============================================
# TEST 4: Check MelSpectrogram transform details
# ============================================
logger.info("\n" + "="*60)
logger.info("TEST 4: MelSpectrogram Transform Details")
logger.info("="*60)

mel_transform = our_frontend.mel_spectrogram
logger.info(f"MelSpectrogram type: {type(mel_transform)}")
logger.info(f"Sample rate: {mel_transform.sample_rate}")
logger.info(f"n_fft: {mel_transform.n_fft}")
logger.info(f"win_length: {mel_transform.win_length}")
logger.info(f"hop_length: {mel_transform.hop_length}")
logger.info(f"f_min: {mel_transform.f_min}")
logger.info(f"f_max: {mel_transform.f_max}")
logger.info(f"n_mels: {mel_transform.n_mels}")
logger.info(f"power: {mel_transform.power}")

# Check the spectrogram transform inside
if hasattr(mel_transform, 'spectrogram'):
    spec_transform = mel_transform.spectrogram
    logger.info(f"\nSpectrogram transform: {type(spec_transform)}")
    logger.info(f"Spectrogram attributes: {dir(spec_transform)}")

    if hasattr(spec_transform, 'center'):
        logger.info(f"center: {spec_transform.center}")
    if hasattr(spec_transform, 'pad_mode'):
        logger.info(f"pad_mode: {spec_transform.pad_mode}")
    if hasattr(spec_transform, 'normalized'):
        logger.info(f"normalized: {spec_transform.normalized}")
    if hasattr(spec_transform, 'onesided'):
        logger.info(f"onesided: {spec_transform.onesided}")

logger.info("\n" + "="*60)
logger.info("Frontend comparison complete!")
logger.info("="*60)
