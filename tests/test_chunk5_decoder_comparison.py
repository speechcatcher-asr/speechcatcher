#!/usr/bin/env python3
"""Compare decoder outputs at chunk 5 with REAL encoder output.

This is the definitive test - we process chunks 1-5 to get the exact
40-frame encoder buffer, then feed it to BOTH decoders and compare.
"""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("CHUNK 5 DECODER COMPARISON (WITH REAL ENCODER OUTPUT)")
print("="*80)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    rate = wavfile_in.getframerate()
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

speech = raw_audio.astype(np.float32) / 32768.0

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

# Process chunks 1-5 to get encoder buffer with 40 frames
print("\n[3] Processing chunks 1-5 to build encoder buffer...")
chunk_size = 8000

espnet_encoder_states = None
our_encoder_states = None

for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    is_final = False
    speech_tensor = torch.from_numpy(chunk)

    # ESPnet
    espnet_feats, espnet_feats_len, espnet_frontend_states = espnet_s2t.apply_frontend(
        speech_tensor, prev_states=espnet_s2t.frontend_states, is_final=is_final
    )
    if espnet_feats is not None:
        with torch.no_grad():
            espnet_enc, espnet_enc_len, espnet_encoder_states = espnet_s2t.asr_model.encoder(
                espnet_feats, espnet_feats_len,
                prev_states=espnet_encoder_states, is_final=is_final, infer_mode=True,
            )
        espnet_s2t.frontend_states = espnet_frontend_states

    # Ours
    our_feats, our_feats_len, our_frontend_states = our_s2t.apply_frontend(
        speech=speech_tensor, prev_states=our_s2t.frontend_states, is_final=is_final
    )
    if our_feats is not None:
        with torch.no_grad():
            our_enc, our_enc_len, our_encoder_states = our_s2t.model.encoder(
                our_feats, our_feats_len,
                prev_states=our_encoder_states, is_final=is_final, infer_mode=True,
            )
        our_s2t.frontend_states = our_frontend_states

    if chunk_idx == 4:
        # Chunk 5 - we should have 40 frames now
        print(f"\nChunk {chunk_idx+1}:")
        print(f"  ESPnet encoder output: {espnet_enc.shape}")
        print(f"  Our encoder output: {our_enc.shape}")

# Accumulate encoder buffers (simulate what beam search does)
print("\n[4] Building encoder buffers (simulating beam search)...")
espnet_buffer = []
our_buffer = []

espnet_encoder_states = None
our_encoder_states = None

for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    is_final = False
    speech_tensor = torch.from_numpy(chunk)

    # ESPnet
    espnet_feats, espnet_feats_len, espnet_frontend_states = espnet_s2t.apply_frontend(
        speech_tensor, prev_states=espnet_s2t.frontend_states if chunk_idx == 0 else espnet_frontend_states, is_final=is_final
    )
    if espnet_feats is not None:
        with torch.no_grad():
            espnet_enc, espnet_enc_len, espnet_encoder_states = espnet_s2t.asr_model.encoder(
                espnet_feats, espnet_feats_len,
                prev_states=espnet_encoder_states, is_final=is_final, infer_mode=True,
            )
        if espnet_enc.size(1) > 0:
            espnet_buffer.append(espnet_enc)

    # Ours
    our_feats, our_feats_len, our_frontend_states = our_s2t.apply_frontend(
        speech=speech_tensor, prev_states=our_s2t.frontend_states if chunk_idx == 0 else our_frontend_states, is_final=is_final
    )
    if our_feats is not None:
        with torch.no_grad():
            our_enc, our_enc_len, our_encoder_states = our_s2t.model.encoder(
                our_feats, our_feats_len,
                prev_states=our_encoder_states, is_final=is_final, infer_mode=True,
            )
        if our_enc.size(1) > 0:
            our_buffer.append(our_enc)

# Concatenate buffers
espnet_encoder_out = torch.cat(espnet_buffer, dim=1)
our_encoder_out = torch.cat(our_buffer, dim=1)

print(f"\nEncoder buffers after chunk 5:")
print(f"  ESPnet: {espnet_encoder_out.shape}")
print(f"  Ours: {our_encoder_out.shape}")

# Extract 40-frame block (what beam search uses)
espnet_block = espnet_encoder_out.narrow(1, 0, 40)
our_block = our_encoder_out.narrow(1, 0, 40)

print(f"\nExtracted 40-frame blocks:")
print(f"  ESPnet: {espnet_block.shape}")
print(f"  Ours: {our_block.shape}")

# Compare encoder outputs
encoder_diff = (espnet_block - our_block).abs().max().item()
print(f"  Max difference: {encoder_diff:.10f}")
if encoder_diff < 0.0001:
    print("  ✅ Encoder outputs MATCH")
else:
    print(f"  ❌ Encoder outputs DIFFER by {encoder_diff}")

# Now test decoders with this EXACT encoder output
print("\n" + "="*80)
print("DECODER COMPARISON WITH 40-FRAME ENCODER OUTPUT")
print("="*80)

# Use ESPnet's encoder output for BOTH (to eliminate encoder differences)
encoder_out = espnet_block
encoder_out_lens = torch.tensor([40], dtype=torch.long)

print(f"\nUsing encoder_out: {encoder_out.shape}")
print(f"Starting hypothesis: [1] (SOS only)\n")

# ============================================================================
# ESPnet Decoder
# ============================================================================
print("[ESPnet Decoder]")

# Get CTC logits
with torch.no_grad():
    espnet_ctc_logits = espnet_s2t.asr_model.ctc.ctc_lo(encoder_out)
print(f"  CTC logits: {espnet_ctc_logits.shape}")

# Get decoder logits for first step
# ESPnet decoder: forward_one_step(tgt, tgt_mask, memory, cache)
ys = torch.tensor([[1]], dtype=torch.long)  # SOS token
tgt_mask = None  # No mask for single token
with torch.no_grad():
    espnet_decoder_out, espnet_cache = espnet_s2t.asr_model.decoder.forward_one_step(
        ys, tgt_mask, encoder_out, cache=None
    )
print(f"  Decoder output: {espnet_decoder_out.shape}")
print(f"  Cache: {type(espnet_cache)}")

# Get log probabilities
# forward_one_step already returns (batch, vocab_size)
espnet_decoder_logits = espnet_decoder_out  # (1, vocab_size)
espnet_decoder_log_probs = torch.log_softmax(espnet_decoder_logits, dim=-1)

print(f"  Decoder log probs: {espnet_decoder_log_probs.shape}")
print(f"    Min: {espnet_decoder_log_probs.min():.4f}")
print(f"    Max: {espnet_decoder_log_probs.max():.4f}")
print(f"    Mean: {espnet_decoder_log_probs.mean():.4f}")

# Top 10 decoder predictions
decoder_top_scores, decoder_top_tokens = espnet_decoder_log_probs[0].topk(10)
print(f"\n  Top 10 decoder predictions:")
for i, (score, token) in enumerate(zip(decoder_top_scores.tolist(), decoder_top_tokens.tolist())):
    print(f"    {i+1}. Token {token:4d}: {score:>8.4f}")

# ============================================================================
# Our Decoder
# ============================================================================
print("\n[Our Decoder]")

# Get CTC logits
with torch.no_grad():
    our_ctc_logits = our_s2t.model.ctc.ctc_lo(encoder_out)
print(f"  CTC logits: {our_ctc_logits.shape}")

# Compare CTC logits
ctc_diff = (espnet_ctc_logits - our_ctc_logits).abs().max().item()
print(f"  CTC logits diff: {ctc_diff:.10f}")
if ctc_diff < 0.0001:
    print("    ✅ CTC logits MATCH")
else:
    print(f"    ❌ CTC logits DIFFER by {ctc_diff}")

# Get decoder logits for first step
ys = torch.tensor([[1]], dtype=torch.long)  # SOS token
tgt_mask = None  # No mask for single token
with torch.no_grad():
    our_decoder_out, our_cache = our_s2t.model.decoder.forward_one_step(
        ys, tgt_mask, encoder_out, cache=None
    )
print(f"  Decoder output: {our_decoder_out.shape}")
print(f"  Cache: {type(our_cache)}")

# Get log probabilities
# forward_one_step already returns (batch, vocab_size)
our_decoder_logits = our_decoder_out  # (1, vocab_size)
our_decoder_log_probs = torch.log_softmax(our_decoder_logits, dim=-1)

print(f"  Decoder log probs: {our_decoder_log_probs.shape}")
print(f"    Min: {our_decoder_log_probs.min():.4f}")
print(f"    Max: {our_decoder_log_probs.max():.4f}")
print(f"    Mean: {our_decoder_log_probs.mean():.4f}")

# Top 10 decoder predictions
our_decoder_top_scores, our_decoder_top_tokens = our_decoder_log_probs[0].topk(10)
print(f"\n  Top 10 decoder predictions:")
for i, (score, token) in enumerate(zip(our_decoder_top_scores.tolist(), our_decoder_top_tokens.tolist())):
    print(f"    {i+1}. Token {token:4d}: {score:>8.4f}")

# Compare decoder logits
decoder_diff = (espnet_decoder_log_probs - our_decoder_log_probs).abs().max().item()
print(f"\n  Decoder log probs diff: {decoder_diff:.10f}")
if decoder_diff < 0.0001:
    print("    ✅ Decoder log probs MATCH")
else:
    print(f"    ❌ Decoder log probs DIFFER by {decoder_diff}")

# ============================================================================
# Compare Top Predictions
# ============================================================================
print("\n" + "="*80)
print("TOP PREDICTION COMPARISON")
print("="*80)

print(f"\nESPnet top token: {decoder_top_tokens[0].item()} (score: {decoder_top_scores[0].item():.4f})")
print(f"Our top token:    {our_decoder_top_tokens[0].item()} (score: {our_decoder_top_scores[0].item():.4f})")

if decoder_top_tokens[0] == our_decoder_top_tokens[0]:
    print("\n✅ TOP TOKENS MATCH!")
else:
    print(f"\n❌ TOP TOKENS DIFFER!")
    print(f"  ESPnet: {decoder_top_tokens[0].item()}")
    print(f"  Ours:   {our_decoder_top_tokens[0].item()}")

# Load BPE to decode
try:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load("/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/data/de_token_list/bpe_unigram1024/bpe.model")

    print(f"\n[Decoded tokens]")
    print(f"ESPnet top 10:")
    for i, token in enumerate(decoder_top_tokens[:10].tolist()):
        piece = sp.IdToPiece(token)
        print(f"  {i+1}. {token:4d} '{piece}'")

    print(f"\nOur top 10:")
    for i, token in enumerate(our_decoder_top_tokens[:10].tolist()):
        piece = sp.IdToPiece(token)
        print(f"  {i+1}. {token:4d} '{piece}'")
except:
    pass

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
