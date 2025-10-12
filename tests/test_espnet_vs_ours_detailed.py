#!/usr/bin/env python3
"""Detailed comparison: espnet_streaming_decoder vs our implementation."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("ESPNET STREAMING DECODER VS OUR IMPLEMENTATION")
print("DETAILED COMPARISON")
print("="*80)

# ============================================================================
# Load audio
# ============================================================================
print("\n[1] Loading audio...")
from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    rate = wavfile_in.getframerate()
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

# Normalize to [-1, 1]
speech = raw_audio.astype(np.float32) / 32768.0

print(f"Audio: rate={rate} Hz, samples={len(speech)}, range=[{speech.min():.4f}, {speech.max():.4f}]")

# ============================================================================
# Load ESPnet streaming decoder
# ============================================================================
print("\n[2] Loading ESPnet streaming decoder...")

from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming as ESPnetStreaming

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_s2t = ESPnetStreaming(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,
    ctc_weight=0.3,
    disable_repetition_detection=False,
)

print("✅ ESPnet streaming decoder loaded")
print(f"   Block size: {espnet_s2t.beam_search.block_size}")
print(f"   Hop size: {espnet_s2t.beam_search.hop_size}")
print(f"   Look ahead: {espnet_s2t.beam_search.look_ahead}")
print(f"   Repetition detection: {not espnet_s2t.beam_search.disable_repetition_detection}")

# ============================================================================
# Load our implementation
# ============================================================================
print("\n[3] Loading our implementation...")

from speechcatcher.speechcatcher import load_model, tags

our_s2t = load_model(
    tags['de_streaming_transformer_xl'],
    beam_size=5,
    quiet=True
)

print("✅ Our implementation loaded")
print(f"   BBD enabled: {our_s2t.beam_search.use_bbd}")
print(f"   BBD conservative: {our_s2t.beam_search.bbd_conservative}")

# ============================================================================
# Process FIRST CHUNK with both implementations
# ============================================================================
print("\n" + "="*80)
print("PROCESSING FIRST CHUNK")
print("="*80)

# Use chunk size that gives us ~40 encoder frames
chunk_size = 25600  # 1.6s at 16kHz
chunk1 = speech[:chunk_size]

print(f"\nChunk 1: {len(chunk1)} samples ({len(chunk1)/rate:.2f}s)")

# ============================================================================
# ESPnet: Process chunk 1
# ============================================================================
print("\n--- ESPnet Processing ---")

espnet_s2t.reset()
speech_tensor = torch.from_numpy(chunk1)

# Step 1: Frontend
print("\n[ESPnet Frontend]")
with torch.no_grad():
    espnet_feats, espnet_feats_len, espnet_frontend_states = espnet_s2t.apply_frontend(
        speech_tensor,
        prev_states=None,
        is_final=False
    )

if espnet_feats is not None:
    print(f"  Features: {espnet_feats.shape}")
    print(f"  Stats: min={espnet_feats.min():.4f}, max={espnet_feats.max():.4f}, mean={espnet_feats.mean():.4f}")
else:
    print(f"  Features: None (waiting for more input)")

# Step 2: Encoder
if espnet_feats is not None:
    print("\n[ESPnet Encoder]")
    with torch.no_grad():
        espnet_enc, espnet_enc_len, espnet_enc_states = espnet_s2t.asr_model.encoder(
            espnet_feats,
            espnet_feats_len,
            prev_states=None,
            is_final=False,
            infer_mode=True,
        )

    print(f"  Encoder output: {espnet_enc.shape}")
    if espnet_enc.numel() > 0:
        print(f"  Stats: min={espnet_enc.min():.4f}, max={espnet_enc.max():.4f}, mean={espnet_enc.mean():.4f}")
    else:
        print(f"  Stats: EMPTY (encoder waiting for more blocks)")
    print(f"  Encoder states: {type(espnet_enc_states)}")

    # Step 3: Beam search
    print("\n[ESPnet Beam Search]")
    with torch.no_grad():
        espnet_results = espnet_s2t.beam_search(
            x=espnet_enc[0],
            maxlenratio=0.0,
            minlenratio=0.0,
            is_final=False,
        )

    print(f"  Beam search processed_block: {espnet_s2t.beam_search.processed_block}")
    print(f"  Beam search process_idx: {espnet_s2t.beam_search.process_idx}")
    if espnet_s2t.beam_search.running_hyps:
        print(f"  Running hypotheses: {len(espnet_s2t.beam_search.running_hyps.yseq)} in batch")
        for i in range(min(3, len(espnet_s2t.beam_search.running_hyps.yseq))):
            yseq = espnet_s2t.beam_search.running_hyps.yseq[i].tolist()
            score = espnet_s2t.beam_search.running_hyps.scores[i].item()
            print(f"    Hyp {i}: {yseq} (score={score:.2f})")

    espnet_assembled = espnet_s2t.assemble_hyps(espnet_results)
    if espnet_assembled:
        text, tokens, token_ids, token_pos, hyp = espnet_assembled[0]
        print(f"\n  ✅ Result: '{text}'")
        print(f"  Tokens: {tokens[:20]}")
        print(f"  Token IDs: {token_ids[:20]}")
    else:
        print(f"\n  ⚠️  No results yet")

# ============================================================================
# Our implementation: Process chunk 1
# ============================================================================
print("\n--- Our Implementation Processing ---")

our_s2t.reset()
speech_tensor = torch.from_numpy(chunk1)

# Step 1: Frontend
print("\n[Our Frontend]")
with torch.no_grad():
    our_feats, our_feats_len, our_frontend_states = our_s2t.apply_frontend(
        speech=speech_tensor,
        prev_states=None,
        is_final=False
    )

if our_feats is not None:
    print(f"  Features: {our_feats.shape}")
    print(f"  Stats: min={our_feats.min():.4f}, max={our_feats.max():.4f}, mean={our_feats.mean():.4f}")
else:
    print(f"  Features: None (waiting for more input)")

# Step 2: Compare features
if espnet_feats is not None and our_feats is not None:
    print("\n[Feature Comparison]")
    feat_diff = (espnet_feats - our_feats).abs()
    print(f"  Max diff: {feat_diff.max():.6f}")
    print(f"  Mean diff: {feat_diff.mean():.6f}")
    if torch.allclose(espnet_feats, our_feats, atol=1e-4):
        print(f"  ✅ Features MATCH")
    else:
        print(f"  ❌ Features DIFFER")

# Step 3: Encoder
if our_feats is not None:
    print("\n[Our Encoder]")
    with torch.no_grad():
        our_enc, our_enc_len, our_enc_states = our_s2t.model.encoder(
            our_feats,
            our_feats_len,
            prev_states=None,
            is_final=False,
            infer_mode=True,
        )

    print(f"  Encoder output: {our_enc.shape}")
    if our_enc.numel() > 0:
        print(f"  Stats: min={our_enc.min():.4f}, max={our_enc.max():.4f}, mean={our_enc.mean():.4f}")
    else:
        print(f"  Stats: EMPTY (encoder waiting for more blocks)")
    print(f"  Encoder states: {type(our_enc_states)}")

    # Compare encoder outputs
    if espnet_enc is not None:
        print("\n[Encoder Comparison]")
        if espnet_enc.numel() > 0 and our_enc.numel() > 0:
            enc_diff = (espnet_enc - our_enc).abs()
            print(f"  Max diff: {enc_diff.max():.6f}")
            print(f"  Mean diff: {enc_diff.mean():.6f}")
            if torch.allclose(espnet_enc, our_enc, atol=1e-4):
                print(f"  ✅ Encoder outputs MATCH")
            else:
                print(f"  ❌ Encoder outputs DIFFER")
        elif espnet_enc.numel() == 0 and our_enc.numel() == 0:
            print(f"  ✅ Both encoders waiting (both empty)")
        else:
            print(f"  ❌ MISMATCH: ESPnet={espnet_enc.shape}, Ours={our_enc.shape}")
            print(f"      One encoder is producing output while the other isn't!")

    # Step 4: Beam search
    print("\n[Our Beam Search]")
    with torch.no_grad():
        our_results = our_s2t(speech=chunk1, is_final=False)

    if our_s2t.beam_state:
        print(f"  Beam state output_index: {our_s2t.beam_state.output_index}")
        print(f"  Hypotheses: {len(our_s2t.beam_state.hypotheses)}")
        for i in range(min(3, len(our_s2t.beam_state.hypotheses))):
            hyp = our_s2t.beam_state.hypotheses[i]
            print(f"    Hyp {i}: {hyp.yseq.tolist()} (score={hyp.score:.2f})")

    if our_results and len(our_results) > 0:
        text, tokens, token_ids = our_results[0]
        print(f"\n  ✅ Result: '{text}'")
        print(f"  Tokens: {tokens[:20]}")
        print(f"  Token IDs: {token_ids[:20]}")
    else:
        print(f"\n  ⚠️  No results yet")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nESPnet streaming decoder:")
if espnet_assembled:
    text, _, _, _, _ = espnet_assembled[0]
    print(f"  Text: '{text}'")
else:
    print(f"  No output yet")

print("\nOur implementation:")
if our_results and len(our_results) > 0:
    text, _, _ = our_results[0]
    print(f"  Text: '{text}'")
else:
    print(f"  No output yet")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
