#!/usr/bin/env python3
"""Check ESPnet's actual runtime beam search configuration."""

import torch

print("="*80)
print("ESPnet RUNTIME CONFIGURATION")
print("="*80)

from espnet_streaming_decoder.asr_inference_streaming import Speech2TextStreaming as ESPnetStreaming

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

print("\n[Loading ESPnet with beam_size=5, ctc_weight=0.3...]")
espnet_s2t = ESPnetStreaming(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
    beam_size=5,
    ctc_weight=0.3,
)

print("✅ Loaded\n")

# Check beam search configuration
if hasattr(espnet_s2t, 'beam_search'):
    bs = espnet_s2t.beam_search
    print("Beam Search Object:")
    print(f"  Type: {type(bs).__name__}")
    print(f"  beam_size: {getattr(bs, 'beam_size', 'N/A')}")
    print(f"  sos: {getattr(bs, 'sos', 'N/A')}")
    print(f"  eos: {getattr(bs, 'eos', 'N/A')}")
    
    if hasattr(bs, 'weights'):
        print(f"\n  Scorer Weights:")
        for name, weight in bs.weights.items():
            print(f"    {name}: {weight}")
    
    if hasattr(bs, 'scorers'):
        print(f"\n  Scorers:")
        for name, scorer in bs.scorers.items():
            print(f"    {name}: {type(scorer).__name__}")

# Check if it's blockwise synchronous
print("\nBlockwise Settings:")
print(f"  block_size: {getattr(bs, 'block_size', 'N/A')}")
print(f"  hop_size: {getattr(bs, 'hop_size', 'N/A')}")
print(f"  look_ahead: {getattr(bs, 'look_ahead', 'N/A')}")

# Check BBD settings
if hasattr(bs, 'use_bbd'):
    print(f"\nBBD Settings:")
    print(f"  use_bbd: {bs.use_bbd}")
    if hasattr(bs, 'bbd_conservative'):
        print(f"  bbd_conservative: {bs.bbd_conservative}")

print("\n" + "="*80)
