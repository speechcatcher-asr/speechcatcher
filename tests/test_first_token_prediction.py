#!/usr/bin/env python3
"""Test what the FIRST token prediction is after SOS=1023."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("FIRST TOKEN PREDICTION (after SOS=1023)")
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

# Process chunks 1-5 to build encoder output
print("\n[3] Building encoder output from chunks 1-5...")
chunk_size = 8000

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

# Concatenate and extract 40-frame block
espnet_encoder_out = torch.cat(espnet_buffer, dim=1).narrow(1, 0, 40)
our_encoder_out = torch.cat(our_buffer, dim=1).narrow(1, 0, 40)

print(f"Encoder output shape: {espnet_encoder_out.shape}")

# Test FIRST token prediction with SOS=1023
print("\n" + "="*80)
print("DECODER PREDICTIONS (starting from SOS=1023)")
print("="*80)

ys = torch.tensor([[1023]], dtype=torch.long)  # SOS=1023 (correct!)
encoder_out = espnet_encoder_out  # Use same encoder output for both

# ESPnet
print("\n[ESPnet]")
with torch.no_grad():
    espnet_decoder_out, _ = espnet_s2t.asr_model.decoder.forward_one_step(
        ys, None, encoder_out, cache=None
    )
espnet_log_probs = torch.log_softmax(espnet_decoder_out, dim=-1)
espnet_top_scores, espnet_top_tokens = espnet_log_probs[0].topk(10)

print("Top 10 predictions:")
for i, (score, token) in enumerate(zip(espnet_top_scores.tolist(), espnet_top_tokens.tolist())):
    piece = sp.IdToPiece(token)
    print(f"  {i+1}. Token {token:4d} '{piece}': {score:>8.4f}")

# Ours
print("\n[Ours]")
with torch.no_grad():
    our_decoder_out, _ = our_s2t.model.decoder.forward_one_step(
        ys, None, encoder_out, cache=None
    )
our_log_probs = torch.log_softmax(our_decoder_out, dim=-1)
our_top_scores, our_top_tokens = our_log_probs[0].topk(10)

print("Top 10 predictions:")
for i, (score, token) in enumerate(zip(our_top_scores.tolist(), our_top_tokens.tolist())):
    piece = sp.IdToPiece(token)
    print(f"  {i+1}. Token {token:4d} '{piece}': {score:>8.4f}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

diff = (espnet_log_probs - our_log_probs).abs().max().item()
print(f"\nMax log prob difference: {diff:.10f}")

if espnet_top_tokens[0] == our_top_tokens[0]:
    print(f"✅ Top tokens MATCH: {espnet_top_tokens[0].item()} '{sp.IdToPiece(espnet_top_tokens[0].item())}'")
else:
    print(f"❌ Top tokens DIFFER:")
    print(f"  ESPnet: {espnet_top_tokens[0].item()} '{sp.IdToPiece(espnet_top_tokens[0].item())}'")
    print(f"  Ours:   {our_top_tokens[0].item()} '{sp.IdToPiece(our_top_tokens[0].item())}'")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
