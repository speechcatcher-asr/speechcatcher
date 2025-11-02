#!/usr/bin/env python3
"""Test encoder and decoder with one block to validate weight loading."""

import logging
import sys
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Load audio and extract features (with normalization)
from speechcatcher.speechcatcher import load_model, convert_inputfile
import wave
import hashlib
import os

logger.info("="*60)
logger.info("ENCODER/DECODER WEIGHT VALIDATION TEST")
logger.info("="*60)

# Load model
logger.info("\nLoading model...")
speech2text = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)

logger.info(f"Model type: {type(speech2text.model)}")
logger.info(f"Model device: {next(speech2text.model.parameters()).device}")

# Check architecture parameters
logger.info("\n" + "="*60)
logger.info("ARCHITECTURE PARAMETERS")
logger.info("="*60)

if hasattr(speech2text.model, 'encoder'):
    encoder = speech2text.model.encoder
    logger.info(f"Encoder type: {type(encoder)}")

    # Try to access encoder config
    if hasattr(encoder, 'encoders'):
        logger.info(f"Number of encoder layers: {len(encoder.encoders)}")

        # Check first layer
        if len(encoder.encoders) > 0:
            first_layer = encoder.encoders[0]
            logger.info(f"First encoder layer type: {type(first_layer)}")

            # Check attention
            if hasattr(first_layer, 'self_attn'):
                attn = first_layer.self_attn
                logger.info(f"Self-attention type: {type(attn)}")

                # Check number of heads
                if hasattr(attn, 'h'):
                    logger.info(f"Number of attention heads: {attn.h}")
                if hasattr(attn, 'linear_q'):
                    logger.info(f"Query linear shape: {attn.linear_q.weight.shape}")
                if hasattr(attn, 'linear_k'):
                    logger.info(f"Key linear shape: {attn.linear_k.weight.shape}")
                if hasattr(attn, 'linear_v'):
                    logger.info(f"Value linear shape: {attn.linear_v.weight.shape}")

# Load audio
logger.info("\n" + "="*60)
logger.info("LOADING AUDIO")
logger.info("="*60)

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

waveform = torch.from_numpy(raw_audio).float()
logger.info(f"Audio: shape={waveform.shape}, dtype={waveform.dtype}")

# Extract features with proper normalization (using apply_frontend)
logger.info("\n" + "="*60)
logger.info("FEATURE EXTRACTION (with normalization)")
logger.info("="*60)

# apply_frontend expects 1D tensor (samples,)
with torch.no_grad():
    # Use apply_frontend to get normalized features
    # This mimics the actual inference pipeline
    features, lengths, _ = speech2text.apply_frontend(
        speech=waveform,  # 1D tensor (samples,)
        prev_states=None,
        is_final=True
    )

if features is not None:
    logger.info(f"Features (normalized): shape={features.shape}")
    logger.info(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}, std={features.std():.4f}")
    logger.info(f"Feature lengths: {lengths}")
else:
    logger.error("Features is None!")
    sys.exit(1)

# TEST 1: Encoder Output
logger.info("\n" + "="*60)
logger.info("TEST 1: ENCODER OUTPUT")
logger.info("="*60)

with torch.no_grad():
    encoder_out, encoder_out_lens, encoder_states = speech2text.model.encoder(
        features,
        lengths,
        prev_states=None,
        is_final=True,
        infer_mode=True
    )

logger.info(f"Encoder output: shape={encoder_out.shape}")
logger.info(f"Encoder output lengths: {encoder_out_lens}")
logger.info(f"Encoder stats: min={encoder_out.min():.4f}, max={encoder_out.max():.4f}, mean={encoder_out.mean():.4f}, std={encoder_out.std():.4f}")

# Check for issues
if torch.isnan(encoder_out).any():
    logger.error("❌ Encoder output contains NaN!")
if torch.isinf(encoder_out).any():
    logger.error("❌ Encoder output contains Inf!")
if not torch.isnan(encoder_out).any() and not torch.isinf(encoder_out).any():
    logger.info("✅ Encoder output is clean (no NaN/Inf)")

# Check value distribution
logger.info("\nEncoder output value distribution:")
logger.info(f"  < -1.0: {(encoder_out < -1.0).sum().item()} values")
logger.info(f"  [-1.0, 0): {((encoder_out >= -1.0) & (encoder_out < 0)).sum().item()} values")
logger.info(f"  [0, 1.0): {((encoder_out >= 0) & (encoder_out < 1.0)).sum().item()} values")
logger.info(f"  >= 1.0: {(encoder_out >= 1.0).sum().item()} values")

# TEST 2: Decoder Output
logger.info("\n" + "="*60)
logger.info("TEST 2: DECODER OUTPUT")
logger.info("="*60)

# Create initial decoder input (SOS token)
sos_id = 1  # Start-of-sequence token
eos_id = 2  # End-of-sequence token

# Initial sequence: just SOS
yseq = torch.tensor([[sos_id]], dtype=torch.long)  # (1, 1)

logger.info(f"Initial decoder input: {yseq}")

with torch.no_grad():
    # Decoder forward pass
    decoder_out, decoder_state = speech2text.model.decoder.score(
        yseq[0],  # (1,) - single sequence
        state=None,  # Initial state
        x=encoder_out[0]  # (enc_len, dim) - encoder output for first sample
    )

logger.info(f"Decoder output: shape={decoder_out.shape}")
logger.info(f"Decoder stats: min={decoder_out.min():.4f}, max={decoder_out.max():.4f}, mean={decoder_out.mean():.4f}, std={decoder_out.std():.4f}")

# Check for issues
if torch.isnan(decoder_out).any():
    logger.error("❌ Decoder output contains NaN!")
if torch.isinf(decoder_out).any():
    logger.error("❌ Decoder output contains Inf!")
if not torch.isnan(decoder_out).any() and not torch.isinf(decoder_out).any():
    logger.info("✅ Decoder output is clean (no NaN/Inf)")

# Get top predictions
top_k = 10
top_scores, top_tokens = torch.topk(decoder_out, k=top_k)

logger.info(f"\nTop {top_k} predictions from decoder:")
for i, (score, token) in enumerate(zip(top_scores.tolist(), top_tokens.tolist())):
    # Try to decode token if tokenizer available
    if speech2text.tokenizer is not None:
        try:
            token_text = speech2text.tokenizer.id_to_piece(int(token))
            logger.info(f"  {i+1}. Token {token:4d} ({token_text:20s}): score={score:.4f}")
        except:
            logger.info(f"  {i+1}. Token {token:4d}: score={score:.4f}")
    else:
        logger.info(f"  {i+1}. Token {token:4d}: score={score:.4f}")

# TEST 3: CTC Output
logger.info("\n" + "="*60)
logger.info("TEST 3: CTC OUTPUT")
logger.info("="*60)

if hasattr(speech2text.model, 'ctc') and speech2text.model.ctc is not None:
    with torch.no_grad():
        ctc_logits = speech2text.model.ctc.ctc_lo(encoder_out)  # (1, enc_len, vocab)
        ctc_probs = torch.softmax(ctc_logits, dim=-1)

    logger.info(f"CTC logits: shape={ctc_logits.shape}")
    logger.info(f"CTC stats: min={ctc_logits.min():.4f}, max={ctc_logits.max():.4f}, mean={ctc_logits.mean():.4f}")

    # Check for issues
    if torch.isnan(ctc_logits).any():
        logger.error("❌ CTC output contains NaN!")
    if torch.isinf(ctc_logits).any():
        logger.error("❌ CTC output contains Inf!")
    if not torch.isnan(ctc_logits).any() and not torch.isinf(ctc_logits).any():
        logger.info("✅ CTC output is clean (no NaN/Inf)")

    # Get most probable tokens at first frame
    frame_idx = 0
    frame_probs = ctc_probs[0, frame_idx]  # (vocab,)
    top_scores, top_tokens = torch.topk(frame_probs, k=top_k)

    logger.info(f"\nTop {top_k} CTC predictions at frame {frame_idx}:")
    for i, (score, token) in enumerate(zip(top_scores.tolist(), top_tokens.tolist())):
        if speech2text.tokenizer is not None:
            try:
                token_text = speech2text.tokenizer.id_to_piece(int(token))
                logger.info(f"  {i+1}. Token {token:4d} ({token_text:20s}): prob={score:.4f}")
            except:
                logger.info(f"  {i+1}. Token {token:4d}: prob={score:.4f}")
        else:
            logger.info(f"  {i+1}. Token {token:4d}: prob={score:.4f}")
else:
    logger.warning("Model does not have CTC module")

# TEST 4: Check for suspicious patterns
logger.info("\n" + "="*60)
logger.info("TEST 4: PATTERN ANALYSIS")
logger.info("="*60)

# Check if decoder always predicts same token
if decoder_out.max() == decoder_out.min():
    logger.error("❌ Decoder output is CONSTANT (all same value)!")
else:
    logger.info("✅ Decoder output has variation")

# Check if top prediction dominates
max_prob = torch.softmax(decoder_out, dim=-1).max().item()
if max_prob > 0.99:
    logger.warning(f"⚠️  Top prediction has very high probability: {max_prob:.4f}")
elif max_prob > 0.9:
    logger.info(f"Top prediction probability: {max_prob:.4f} (high but reasonable)")
else:
    logger.info(f"Top prediction probability: {max_prob:.4f}")

# Check top token
top_token = top_tokens[0].item()
if top_token == 1023:
    logger.warning(f"⚠️  Top prediction is token 1023 (the repetitive م token from tests)")
elif top_token in [3, 4]:
    logger.warning(f"⚠️  Top prediction is token {top_token} (punctuation from CTC tests)")
else:
    logger.info(f"Top prediction token: {top_token}")

logger.info("\n" + "="*60)
logger.info("WEIGHT VALIDATION TEST COMPLETE")
logger.info("="*60)
