#!/usr/bin/env python3
"""Compare exact decoder + CTC scores for tokens 372 and 738."""

import torch
import numpy as np
import wave
import hashlib
import os
import json

print("="*80)
print("EXACT SCORE COMPARISON (Tokens 372 vs 738)")
print("="*80)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as f:
    raw_audio = np.frombuffer(f.readframes(-1), dtype='int16')
speech = raw_audio.astype(np.float32) / 32768.0

# Load token list
with open("/tmp/espnet_token_list.json", "r") as f:
    token_list = json.load(f)

# Load ESPnet
print("\n[1] Loading ESPnet...")
from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming as ESPnetStreaming

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_s2t = ESPnetStreaming(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,
    ctc_weight=0.3,
)
espnet_s2t.reset()
print("✅ ESPnet loaded")

# Load ours
print("\n[2] Loading our implementation...")
from speechcatcher.speechcatcher import load_model, tags

our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()
print("✅ Ours loaded")

# Process chunks 1-5 to get to first decoding point
print("\n[3] Processing chunks 1-5...")
chunk_size = 8000

for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    espnet_s2t(chunk, is_final=False)
    our_s2t(chunk, is_final=False)

print("✅ Processed chunks 1-5")

# Now we should have 40-frame encoder buffer and be ready to decode block 0
print("\n[4] Extracting 40-frame encoder output...")

# Get SAME 40 frames from both
espnet_enc = espnet_s2t.beam_search.encbuffer[:40].unsqueeze(0)  # (1, 40, 256)
our_enc = our_s2t.beam_search.encoder_buffer[:, :40, :]  # (1, 40, 256)

print(f"ESPnet encoder: {espnet_enc.shape}")
print(f"Our encoder: {our_enc.shape}")
print(f"Match: {torch.allclose(espnet_enc, our_enc, atol=1e-6)}")

# Use SAME encoder output for both
enc = espnet_enc

print("\n" + "="*80)
print("DECODER SCORES (from SOS=1023)")
print("="*80)

# Initial hypothesis: [SOS]
ys = torch.tensor([[1023]], dtype=torch.long)

# ESPnet decoder
with torch.no_grad():
    espnet_dec_out, _ = espnet_s2t.asr_model.decoder.forward_one_step(ys, None, enc, cache=None)
espnet_dec_logprobs = torch.log_softmax(espnet_dec_out, dim=-1)[0]

print(f"\nToken 372 (▁Li):    {espnet_dec_logprobs[372].item():.6f}")
print(f"Token 738 (▁liebe): {espnet_dec_logprobs[738].item():.6f}")
print(f"Difference:         {(espnet_dec_logprobs[372] - espnet_dec_logprobs[738]).item():.6f}")

# Our decoder (should be identical)
with torch.no_grad():
    our_dec_out, _ = our_s2t.model.decoder.forward_one_step(ys, None, enc, cache=None)
our_dec_logprobs = torch.log_softmax(our_dec_out, dim=-1)[0]

print(f"\n[Verification - should match ESPnet]")
print(f"Token 372 (▁Li):    {our_dec_logprobs[372].item():.6f}")
print(f"Token 738 (▁liebe): {our_dec_logprobs[738].item():.6f}")

print("\n" + "="*80)
print("CTC SCORES")
print("="*80)

# Get CTC logits for 40 frames
with torch.no_grad():
    espnet_ctc_logits = espnet_s2t.asr_model.ctc.ctc_lo(enc)  # (1, 40, 1024)
espnet_ctc_logprobs = torch.log_softmax(espnet_ctc_logits, dim=-1)[0]  # (40, 1024)

# Simple sum across frames (not true CTC prefix scoring, but gives intuition)
espnet_ctc_sum_372 = espnet_ctc_logprobs[:, 372].sum().item()
espnet_ctc_sum_738 = espnet_ctc_logprobs[:, 738].sum().item()

print(f"\nToken 372 (▁Li)    sum: {espnet_ctc_sum_372:.6f}")
print(f"Token 738 (▁liebe) sum: {espnet_ctc_sum_738:.6f}")
print(f"Difference:             {(espnet_ctc_sum_372 - espnet_ctc_sum_738):.6f}")

print("\n" + "="*80)
print("COMBINED SCORES (decoder=0.7, ctc=0.3)")
print("="*80)

# NOTE: This is simplified - real CTC prefix scoring is more complex
dec_372 = espnet_dec_logprobs[372].item()
dec_738 = espnet_dec_logprobs[738].item()

# Simplified combined score
combined_372 = 0.7 * dec_372 + 0.3 * (espnet_ctc_sum_372 / 40)
combined_738 = 0.7 * dec_738 + 0.3 * (espnet_ctc_sum_738 / 40)

print(f"\nToken 372 (▁Li):")
print(f"  Decoder (0.7x): {0.7 * dec_372:.6f}")
print(f"  CTC (0.3x):     {0.3 * (espnet_ctc_sum_372 / 40):.6f}")
print(f"  Combined:       {combined_372:.6f}")

print(f"\nToken 738 (▁liebe):")
print(f"  Decoder (0.7x): {0.7 * dec_738:.6f}")
print(f"  CTC (0.3x):     {0.3 * (espnet_ctc_sum_738 / 40):.6f}")
print(f"  Combined:       {combined_738:.6f}")

print(f"\nWinner: Token {'372 (▁Li)' if combined_372 > combined_738 else '738 (▁liebe)'}")
print(f"Margin: {abs(combined_372 - combined_738):.6f}")

print("\n" + "="*80)
print("NOTE: CTC scoring here is simplified (sum/frames). Real CTC prefix")
print("scoring uses forward algorithm with proper blank handling.")
print("="*80)
