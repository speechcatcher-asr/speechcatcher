#!/usr/bin/env python3
"""Compare decoder outputs given SAME encoder output."""

import torch
import numpy as np
import wave
import hashlib
import os

print("="*80)
print("DECODER COMPARISON WITH SAME ENCODER OUTPUT")
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

# Process chunks until we have encoder output
print("\n[3] Processing chunks until encoder produces output...")
chunk_size = 8000
espnet_encoder_states = None
our_encoder_states = None
espnet_encoder_buffer = []
our_encoder_buffer = []

for chunk_idx in range(6):
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
        if espnet_enc.size(1) > 0:
            espnet_encoder_buffer.append(espnet_enc)
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
        if our_enc.size(1) > 0:
            our_encoder_buffer.append(our_enc)
        our_s2t.frontend_states = our_frontend_states

    # Check if we have first encoder output
    if len(espnet_encoder_buffer) > 0 and len(our_encoder_buffer) > 0:
        print(f"\nChunk {chunk_idx+1}: Both have encoder outputs!")
        print(f"  ESPnet: {espnet_encoder_buffer[0].shape}")
        print(f"  Ours:   {our_encoder_buffer[0].shape}")

        # Use ESPnet's encoder output for BOTH decoders
        encoder_out = espnet_encoder_buffer[0]
        encoder_out_lens = torch.tensor([encoder_out.size(1)], dtype=torch.long)

        print(f"\n[4] Testing decoder with SAME encoder output: {encoder_out.shape}")

        # ESPnet decoder
        print("\nESPnet beam search...")
        with torch.no_grad():
            espnet_results = espnet_s2t.beam_search(encoder_out, encoder_out_lens)

        espnet_best = espnet_results[0]
        print(f"ESPnet tokens: {espnet_best.yseq.tolist()}")
        print(f"ESPnet score: {espnet_best.score:.4f}")

        # Our decoder (use standard beam search, not streaming)
        print("\nOur beam search...")
        from speechcatcher.beam_search.beam_search import StandardBeamSearch

        our_beam_search = StandardBeamSearch(
            scorers={
                "decoder": our_s2t.model.decoder,
                "ctc": our_s2t.beam_search.scorers["ctc"],
            },
            weights={
                "decoder": 0.7,
                "ctc": 0.3,
            },
            beam_size=5,
            vocab_size=1024,
            sos_id=1,
            eos_id=2,
            device="cpu",
        )

        with torch.no_grad():
            our_results = our_beam_search(encoder_out, encoder_out_lens)

        our_best = our_results[0]
        print(f"Our tokens: {our_best.yseq.tolist()}")
        print(f"Our score: {our_best.score:.4f}")

        # Compare
        if espnet_best.yseq.tolist() == our_best.yseq.tolist():
            print("\n✅ DECODERS PRODUCE SAME OUTPUT!")
        else:
            print("\n❌ DECODERS PRODUCE DIFFERENT OUTPUT!")

        break

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
