#!/usr/bin/env python3
"""Compare our decoder output vs ESPnet's decoder output with same inputs."""

import logging
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("DECODER OUTPUT COMPARISON: Our Implementation vs ESPnet")
logger.info("="*80)

# ============================================================================
# Load our model
# ============================================================================
logger.info("\n[1] Loading OUR model...")
from speechcatcher.speechcatcher import load_model

our_model = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)

logger.info("✅ Our model loaded")

# ============================================================================
# Load ESPnet model directly
# ============================================================================
logger.info("\n[2] Loading ESPnet model...")

from espnet2.bin.asr_inference_streaming import Speech2TextStreaming as ESPnetS2T

espnet_model = ESPnetS2T(
    asr_train_config="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml",
    asr_model_file="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth",
    device="cpu",
)

logger.info("✅ ESPnet model loaded")

# ============================================================================
# Create test inputs
# ============================================================================
logger.info("\n[3] Creating test inputs...")

# Use SAME encoder output for both
encoder_out = torch.randn(1, 124, 256)  # Fixed random encoder output
sos_id = 1
yseq = torch.tensor([sos_id])

logger.info(f"Encoder output: {encoder_out.shape}")
logger.info(f"Input sequence: {yseq}")

# ============================================================================
# Test our decoder
# ============================================================================
logger.info("\n[4] Testing OUR decoder...")

with torch.no_grad():
    our_decoder_out, our_state = our_model.model.decoder.score(
        yseq,
        state=None,
        x=encoder_out[0]  # (124, 256)
    )

logger.info(f"Our decoder output: {our_decoder_out.shape}")
logger.info(f"Our decoder stats: min={our_decoder_out.min():.4f}, max={our_decoder_out.max():.4f}, mean={our_decoder_out.mean():.4f}")

top_k = 5
top_scores, top_tokens = torch.topk(our_decoder_out, k=top_k)
logger.info(f"\nOur top {top_k} predictions:")
for i, (score, token) in enumerate(zip(top_scores.tolist(), top_tokens.tolist())):
    if our_model.tokenizer:
        token_text = our_model.tokenizer.id_to_piece(int(token))
        logger.info(f"  {i+1}. Token {token:4d} ({token_text:20s}): {score:.4f}")

# ============================================================================
# Test ESPnet decoder
# ============================================================================
logger.info(f"\n[5] Testing ESPnet decoder...")

with torch.no_grad():
    # ESPnet decoder.score() has same signature
    espnet_decoder_out, espnet_state = espnet_model.asr_model.decoder.score(
        yseq,
        state=None,
        x=encoder_out[0]  # (124, 256)
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
# Compare outputs
# ============================================================================
logger.info(f"\n[6] Comparing outputs...")

diff = (our_decoder_out - espnet_decoder_out).abs()
logger.info(f"\nAbsolute difference:")
logger.info(f"  Max: {diff.max():.6f}")
logger.info(f"  Mean: {diff.mean():.6f}")
logger.info(f"  Min: {diff.min():.6f}")

if torch.allclose(our_decoder_out, espnet_decoder_out, atol=1e-4):
    logger.info("\n✅ Outputs are IDENTICAL (within tolerance)!")
else:
    logger.error("\n❌ Outputs are DIFFERENT!")

    # Find which tokens differ most
    _, our_top_token = our_decoder_out.topk(1)
    _, espnet_top_token = espnet_decoder_out.topk(1)

    logger.info(f"\nTop prediction:")
    logger.info(f"  Ours: token {our_top_token.item()}")
    logger.info(f"  ESPnet: token {espnet_top_token.item()}")

    if our_top_token.item() == 1023:
        logger.error(f"  ❌ Our model predicts token 1023 (wrong!)")
    if espnet_top_token.item() != our_top_token.item():
        logger.error(f"  ❌ Different top predictions!")

logger.info("\n" + "="*80)
logger.info("COMPARISON COMPLETE")
logger.info("="*80)
