#!/usr/bin/env python3
"""Compare full ASR pipeline: Our implementation vs ESPnet with REAL audio."""

import logging
import torch
import numpy as np
from pathlib import Path
import wave
import hashlib
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("FULL PIPELINE COMPARISON: Our Implementation vs ESPnet")
logger.info("="*80)

# ============================================================================
# Load audio using speechcatcher's ffmpeg function
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
logger.info(f"Audio loaded: {waveform.shape}")

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

espnet_model = ESPnetS2T(
    asr_train_config="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml",
    asr_model_file="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth",
    device="cpu",
)

logger.info("✅ ESPnet model loaded")

# ============================================================================
# Test OUR full pipeline
# ============================================================================
logger.info("\n[4] Testing OUR full pipeline...")

with torch.no_grad():
    # Frontend + Normalize
    our_features, our_lengths, _ = our_model.apply_frontend(
        speech=waveform,
        prev_states=None,
        is_final=True
    )

    logger.info(f"Our features: {our_features.shape}")

    # Encoder
    our_encoder_out, our_encoder_lens, _ = our_model.model.encoder(
        our_features,
        our_lengths,
        prev_states=None,
        is_final=True,
        infer_mode=True
    )

    logger.info(f"Our encoder output: {our_encoder_out.shape}")
    logger.info(f"Our encoder stats: min={our_encoder_out.min():.4f}, max={our_encoder_out.max():.4f}, mean={our_encoder_out.mean():.4f}")

    # Decoder (first token from SOS)
    sos_id = 1
    yseq = torch.tensor([sos_id])

    our_decoder_out, our_state = our_model.model.decoder.score(
        yseq,
        state=None,
        x=our_encoder_out[0]  # (enc_len, 256)
    )

    logger.info(f"Our decoder output: {our_decoder_out.shape}")
    logger.info(f"Our decoder stats: min={our_decoder_out.min():.4f}, max={our_decoder_out.max():.4f}, mean={our_decoder_out.mean():.4f}")

# Get top predictions
top_k = 10
top_scores, top_tokens = torch.topk(our_decoder_out, k=top_k)

logger.info(f"\nOur top {top_k} predictions:")
for i, (score, token) in enumerate(zip(top_scores.tolist(), top_tokens.tolist())):
    if our_model.tokenizer:
        token_text = our_model.tokenizer.id_to_piece(int(token))
        logger.info(f"  {i+1}. Token {token:4d} ({token_text:20s}): {score:.4f}")

# ============================================================================
# Test ESPnet full pipeline
# ============================================================================
logger.info(f"\n[5] Testing ESPnet full pipeline...")

with torch.no_grad():
    # ESPnet uses speech array directly
    speech_array = waveform.numpy()

    # ESPnet's __call__ method processes the full pipeline
    # But we want to test step-by-step, so we'll do it manually

    # Frontend
    speech_batch = waveform.unsqueeze(0)  # (1, samples)
    speech_lengths = torch.tensor([waveform.shape[0]])
    espnet_features, espnet_lengths = espnet_model.asr_model.frontend(speech_batch, speech_lengths)
    logger.info(f"ESPnet features: {espnet_features.shape}")

    # Normalize
    espnet_features, espnet_lengths = espnet_model.asr_model.normalize(espnet_features, espnet_lengths)
    logger.info(f"ESPnet features (normalized): {espnet_features.shape}")

    # Encoder
    espnet_encoder_out, espnet_encoder_lens, _ = espnet_model.asr_model.encoder(
        espnet_features,
        espnet_lengths,
        prev_states=None,
        is_final=True,
        infer_mode=True
    )

    logger.info(f"ESPnet encoder output: {espnet_encoder_out.shape}")
    logger.info(f"ESPnet encoder stats: min={espnet_encoder_out.min():.4f}, max={espnet_encoder_out.max():.4f}, mean={espnet_encoder_out.mean():.4f}")

    # Decoder
    espnet_decoder_out, espnet_state = espnet_model.asr_model.decoder.score(
        yseq,
        state=None,
        x=espnet_encoder_out[0]
    )

    logger.info(f"ESPnet decoder output: {espnet_decoder_out.shape}")
    logger.info(f"ESPnet decoder stats: min={espnet_decoder_out.min():.4f}, max={espnet_decoder_out.max():.4f}, mean={espnet_decoder_out.mean():.4f}")

top_scores, top_tokens = torch.topk(espnet_decoder_out, k=top_k)

logger.info(f"\nESPnet top {top_k} predictions:")
for i, (score, token) in enumerate(zip(top_scores.tolist(), top_tokens.tolist())):
    if our_model.tokenizer:
        token_text = our_model.tokenizer.id_to_piece(int(token))
        logger.info(f"  {i+1}. Token {token:4d} ({token_text:20s}): {score:.4f}")

# ============================================================================
# Compare at each stage
# ============================================================================
logger.info(f"\n[6] Comparing outputs at each stage...")

# Features
features_diff = (our_features - espnet_features).abs()
logger.info(f"\n📊 Features comparison:")
logger.info(f"  Max diff: {features_diff.max():.6f}")
logger.info(f"  Mean diff: {features_diff.mean():.6f}")
if torch.allclose(our_features, espnet_features, atol=1e-5):
    logger.info("  ✅ Features IDENTICAL")
else:
    logger.warning("  ⚠️  Features DIFFER")

# Encoder output
encoder_diff = (our_encoder_out - espnet_encoder_out).abs()
logger.info(f"\n📊 Encoder output comparison:")
logger.info(f"  Max diff: {encoder_diff.max():.6f}")
logger.info(f"  Mean diff: {encoder_diff.mean():.6f}")
if torch.allclose(our_encoder_out, espnet_encoder_out, atol=1e-4):
    logger.info("  ✅ Encoder outputs IDENTICAL")
else:
    logger.warning("  ⚠️  Encoder outputs DIFFER")

# Decoder output
decoder_diff = (our_decoder_out - espnet_decoder_out).abs()
logger.info(f"\n📊 Decoder output comparison:")
logger.info(f"  Max diff: {decoder_diff.max():.6f}")
logger.info(f"  Mean diff: {decoder_diff.mean():.6f}")
if torch.allclose(our_decoder_out, espnet_decoder_out, atol=1e-4):
    logger.info("  ✅ Decoder outputs IDENTICAL")
else:
    logger.warning("  ⚠️  Decoder outputs DIFFER")

# Top token comparison
our_top_token = top_tokens[0].item() if 'top_tokens' in locals() else None
if our_top_token == 1023:
    logger.warning(f"\n⚠️  Top prediction is token 1023 (Arabic م) - UNEXPECTED for German audio!")
    logger.warning("  This suggests the model itself might have an issue, not our implementation.")
else:
    logger.info(f"\n✅ Top prediction token {our_top_token} seems reasonable")

logger.info("\n" + "="*80)
logger.info("FULL PIPELINE COMPARISON COMPLETE")
logger.info("="*80)
