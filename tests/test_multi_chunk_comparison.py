#!/usr/bin/env python3
"""Multi-chunk comparison: ESPnet streaming vs our implementation."""

import torch
import numpy as np
import wave
import hashlib
import os
import logging

logging.basicConfig(level=logging.WARNING, format='%(message)s')

print("="*80)
print("MULTI-CHUNK STREAMING COMPARISON")
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
    disable_repetition_detection=False,
)
espnet_s2t.reset()
print(f"✅ Loaded (block={espnet_s2t.beam_search.block_size}, hop={espnet_s2t.beam_search.hop_size}, lookahead={espnet_s2t.beam_search.look_ahead})")

# Load ours
print("\n[3] Loading our implementation...")
from speechcatcher.speechcatcher import load_model, tags

our_s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)
our_s2t.reset()
print(f"✅ Loaded")

# Process in chunks
print("\n" + "="*80)
print("PROCESSING CHUNKS")
print("="*80)

chunk_size = 8000  # Match typical streaming chunk size
num_chunks = (len(speech) + chunk_size - 1) // chunk_size

print(f"\nChunk size: {chunk_size} samples ({chunk_size/rate:.2f}s)")
print(f"Total chunks: {num_chunks}\n")

for chunk_idx in range(num_chunks):
    start = chunk_idx * chunk_size
    end = min((chunk_idx + 1) * chunk_size, len(speech))
    chunk = speech[start:end]
    is_final = (chunk_idx == num_chunks - 1)

    print(f"--- Chunk {chunk_idx+1}/{num_chunks} (is_final={is_final}) ---")

    # ESPnet
    with torch.no_grad():
        espnet_results = espnet_s2t(chunk, is_final=is_final)

    espnet_text = ""
    if espnet_results:
        espnet_text, _, _, _, _ = espnet_results[0]

    # Check encoder output size
    if hasattr(espnet_s2t, 'encoder_states') and espnet_s2t.encoder_states:
        enc_info = f"enc_states=present"
    else:
        enc_info = f"enc_states=None"

    # Check beam search state
    bs = espnet_s2t.beam_search
    bs_info = f"block={bs.processed_block}, idx={bs.process_idx}"
    if bs.encbuffer is not None:
        bs_info += f", encbuf={bs.encbuffer.shape[0]}"

    print(f"  ESPnet: '{espnet_text[:50]}' ({bs_info}, {enc_info})")

    # Ours
    with torch.no_grad():
        our_results = our_s2t(chunk, is_final=is_final)

    our_text = ""
    if our_results and len(our_results) > 0:
        our_text, _, _ = our_results[0]

    # Check beam state
    if our_s2t.beam_state:
        bs_info = f"output_idx={our_s2t.beam_state.output_index}, hyps={len(our_s2t.beam_state.hypotheses)}"
        if our_s2t.beam_state.encoder_out is not None:
            bs_info += f", enc_out={our_s2t.beam_state.encoder_out.shape[1]}"
    else:
        bs_info = "beam_state=None"

    print(f"  Ours:   '{our_text[:50]}' ({bs_info})")

    # Break if final
    if is_final:
        break

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

# Get final outputs
espnet_s2t.reset()
espnet_final = espnet_s2t(speech, is_final=True)
espnet_final_text = ""
if espnet_final:
    espnet_final_text, _, _, _, _ = espnet_final[0]

our_s2t.reset()
our_final = our_s2t(speech, is_final=True)
our_final_text = ""
if our_final and len(our_final) > 0:
    our_final_text, _, _ = our_final[0]

print(f"\nESPnet (full audio): '{espnet_final_text}'")
print(f"\nOurs (full audio): '{our_final_text}'")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
