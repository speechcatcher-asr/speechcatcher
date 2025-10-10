#!/usr/bin/env python3
"""Compare encoder streaming behavior: ESPnet vs Ours."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("ENCODER STREAMING BEHAVIOR COMPARISON")
print("="*80)

# Load audio
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

speech = raw_audio.astype(np.float32) / 32768.0
print(f"Audio: {len(speech)} samples ({len(speech)/rate:.2f}s)")

# Load ESPnet
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
)
espnet_s2t.reset()
print("✅ ESPnet loaded")

# Load ours
print("\n[3] Loading our implementation...")
from speechcatcher.speechcatcher import load_model, tags

our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()
print("✅ Ours loaded")

# Process chunks and compare ENCODER outputs
print("\n" + "="*80)
print("ENCODER OUTPUT COMPARISON (Chunk by Chunk)")
print("="*80)

chunk_size = 8000
chunks = [
    speech[i*chunk_size : min((i+1)*chunk_size, len(speech))]
    for i in range((len(speech) + chunk_size - 1) // chunk_size)
]

espnet_encoder_states = None
our_encoder_states = None

for chunk_idx, chunk in enumerate(chunks[:6]):  # First 6 chunks
    is_final = (chunk_idx == len(chunks) - 1)

    print(f"\n{'='*60}")
    print(f"CHUNK {chunk_idx+1}/{len(chunks)} ({len(chunk)} samples, is_final={is_final})")
    print(f"{'='*60}")

    # === ESPnet encoder ===
    speech_tensor = torch.from_numpy(chunk)

    # Frontend
    with torch.no_grad():
        espnet_feats, espnet_feats_len, espnet_frontend_states = espnet_s2t.apply_frontend(
            speech_tensor,
            prev_states=espnet_s2t.frontend_states,
            is_final=is_final
        )

    if espnet_feats is not None:
        print(f"\nESPnet Frontend: {espnet_feats.shape}")

        # Encoder
        with torch.no_grad():
            espnet_enc, espnet_enc_len, espnet_encoder_states = espnet_s2t.asr_model.encoder(
                espnet_feats,
                espnet_feats_len,
                prev_states=espnet_encoder_states,
                is_final=is_final,
                infer_mode=True,
            )

        print(f"ESPnet Encoder OUT: {espnet_enc.shape}")
        print(f"  Encoder states type: {type(espnet_encoder_states)}")
        if isinstance(espnet_encoder_states, dict):
            for key in list(espnet_encoder_states.keys())[:3]:
                val = espnet_encoder_states[key]
                if isinstance(val, torch.Tensor):
                    print(f"    {key}: {val.shape}")
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                    print(f"    {key}: list of {len(val)} tensors, first: {val[0].shape}")

        espnet_s2t.frontend_states = espnet_frontend_states
    else:
        print(f"\nESPnet Frontend: None (waiting)")

    # === Our encoder ===
    speech_tensor = torch.from_numpy(chunk)

    # Frontend
    with torch.no_grad():
        our_feats, our_feats_len, our_frontend_states = our_s2t.apply_frontend(
            speech=speech_tensor,
            prev_states=our_s2t.frontend_states,
            is_final=is_final
        )

    if our_feats is not None:
        print(f"\nOur Frontend: {our_feats.shape}")

        # Encoder
        with torch.no_grad():
            our_enc, our_enc_len, our_encoder_states = our_s2t.model.encoder(
                our_feats,
                our_feats_len,
                prev_states=our_encoder_states,
                is_final=is_final,
                infer_mode=True,
            )

        print(f"Our Encoder OUT: {our_enc.shape}")
        print(f"  Encoder states type: {type(our_encoder_states)}")
        if isinstance(our_encoder_states, dict):
            for key in list(our_encoder_states.keys())[:3]:
                val = our_encoder_states[key]
                if isinstance(val, torch.Tensor):
                    print(f"    {key}: {val.shape}")
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                    print(f"    {key}: list of {len(val)} tensors, first: {val[0].shape}")

        our_s2t.frontend_states = our_frontend_states
    else:
        print(f"\nOur Frontend: None (waiting)")

    # Compare
    if espnet_feats is not None and our_feats is not None:
        if espnet_enc.shape == our_enc.shape:
            if espnet_enc.numel() > 0 and our_enc.numel() > 0:
                diff = (espnet_enc - our_enc).abs().max()
                print(f"\n✅ Encoder outputs: SAME SHAPE, max diff={diff:.6f}")
            elif espnet_enc.numel() == 0 and our_enc.numel() == 0:
                print(f"\n✅ Encoder outputs: BOTH EMPTY (waiting for context)")
            else:
                print(f"\n⚠️  Encoder outputs: SAME SHAPE but different sizes?")
        else:
            print(f"\n❌ Encoder outputs: DIFFERENT SHAPES!")
            print(f"   ESPnet: {espnet_enc.shape}")
            print(f"   Ours:   {our_enc.shape}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
