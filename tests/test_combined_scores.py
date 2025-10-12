#!/usr/bin/env python3
"""Test combined decoder + CTC scores to see which token wins."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("COMBINED SCORES TEST (Decoder + CTC)")
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

# Load BPE
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/data/de_token_list/bpe_unigram1024/bpe.model")

# Load our implementation
print("\n[1] Loading model...")
from speechcatcher.speechcatcher import load_model, tags

s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
s2t.reset()
print("✅ Model loaded")

# Process chunks 1-5 to build encoder output
print("\n[2] Building encoder output from chunks 1-5...")
chunk_size = 8000

buffer = []
encoder_states = None

for chunk_idx in range(5):
    chunk = speech[chunk_idx*chunk_size : min((chunk_idx+1)*chunk_size, len(speech))]
    is_final = False
    speech_tensor = torch.from_numpy(chunk)

    feats, feats_len, frontend_states = s2t.apply_frontend(
        speech=speech_tensor, prev_states=s2t.frontend_states if chunk_idx == 0 else frontend_states, is_final=is_final
    )
    if feats is not None:
        with torch.no_grad():
            enc, enc_len, encoder_states = s2t.model.encoder(
                feats, feats_len,
                prev_states=encoder_states, is_final=is_final, infer_mode=True,
            )
        if enc.size(1) > 0:
            buffer.append(enc)

# Extract 40-frame block
encoder_out = torch.cat(buffer, dim=1).narrow(1, 0, 40)
print(f"Encoder output shape: {encoder_out.shape}")

# Get decoder and CTC scores
print("\n[3] Computing decoder and CTC scores...")

# Decoder scores (starting from SOS=1023)
ys = torch.tensor([[1023]], dtype=torch.long)
with torch.no_grad():
    decoder_out, _ = s2t.model.decoder.forward_one_step(ys, None, encoder_out, cache=None)
decoder_log_probs = torch.log_softmax(decoder_out, dim=-1)[0]  # (vocab_size,)

# CTC scores (for initial hypothesis with just SOS)
with torch.no_grad():
    ctc_logits = s2t.model.ctc.ctc_lo(encoder_out)  # (1, 40, vocab_size)
ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)[0]  # (40, vocab_size)

# For CTC prefix scoring, we need to compute the score for each token
# This is complex, but for a quick test, let's just look at the sum of CTC log probs
# across all frames for each token
ctc_sum_scores = ctc_log_probs.sum(dim=0)  # (vocab_size,)

print("\n" + "="*80)
print("SCORE BREAKDOWN FOR KEY TOKENS")
print("="*80)

# Compare key tokens: 372 (▁dieses), 738 (trag), 1023 (م)
key_tokens = [
    (372, '▁dieses'),
    (738, 'trag'),
    (1023, 'م'),
]

decoder_weight = 0.7
ctc_weight = 0.3

print(f"\nWeights: decoder={decoder_weight}, ctc={ctc_weight}")
print(f"\nToken breakdown:\n")

for token_id, token_text in key_tokens:
    dec_score = decoder_log_probs[token_id].item()
    ctc_score = ctc_sum_scores[token_id].item()
    combined = decoder_weight * dec_score + ctc_weight * ctc_score

    print(f"Token {token_id:4d} '{token_text}':")
    print(f"  Decoder score:  {dec_score:>10.4f}  (weighted: {decoder_weight * dec_score:>10.4f})")
    print(f"  CTC sum score:  {ctc_score:>10.4f}  (weighted: {ctc_weight * ctc_score:>10.4f})")
    print(f"  Combined:       {combined:>10.4f}")
    print()

# Find top 10 by combined score
combined_scores = decoder_weight * decoder_log_probs + ctc_weight * ctc_sum_scores
top_scores, top_tokens = combined_scores.topk(10)

print("="*80)
print("TOP 10 BY COMBINED SCORE")
print("="*80)
print()

for i, (score, token) in enumerate(zip(top_scores.tolist(), top_tokens.tolist())):
    piece = sp.IdToPiece(token)
    dec = decoder_log_probs[token].item()
    ctc = ctc_sum_scores[token].item()
    print(f"{i+1:2d}. Token {token:4d} '{piece:12s}': {score:>8.4f}  (dec: {dec:>7.3f}, ctc: {ctc:>8.3f})")

print("\n" + "="*80)
print("NOTE: CTC scores here are just summed log probs, not true CTC prefix scores")
print("Real CTC prefix scoring uses forward algorithm with blank handling")
print("="*80)
