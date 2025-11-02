#!/usr/bin/env python3
"""Compare beam search configuration between ESPnet and ours."""

import torch
import yaml

print("="*80)
print("BEAM SEARCH CONFIGURATION COMPARISON")
print("="*80)

# Load ESPnet config
config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("\n[ESPnet Config]")
print("\nBeam Search Parameters:")
for key in ['beam_size', 'ctc_weight', 'penalty', 'maxlenratio', 'minlenratio']:
    if key in config:
        print(f"  {key}: {config[key]}")

print("\nStreaming Parameters:")
for key in ['streaming', 'block_size', 'hop_size', 'look_ahead']:
    if key in config.get('encoder_conf', {}):
        print(f"  encoder.{key}: {config['encoder_conf'][key]}")

print("\nDecoder Parameters:")
decoder_conf = config.get('decoder_conf', {})
for key in ['attention_heads', 'linear_units', 'num_blocks', 'dropout_rate']:
    if key in decoder_conf:
        print(f"  decoder.{key}: {decoder_conf[key]}")

# Load our implementation
print("\n" + "="*80)
print("[Our Implementation]")
print("="*80)

from speechcatcher.speechcatcher import load_model, tags

s2t = load_model(tags['de_streaming_transformer_xl'], beam_size=5, quiet=True)

print("\nBeam Search Parameters:")
if hasattr(s2t, 'beam_search'):
    bs = s2t.beam_search
    print(f"  beam_size: {bs.beam_size}")
    print(f"  vocab_size: {bs.vocab_size}")
    print(f"  sos_id: {bs.sos_id}")
    print(f"  eos_id: {bs.eos_id}")
    print(f"  block_size: {bs.block_size}")
    print(f"  hop_size: {bs.hop_size}")
    print(f"  look_ahead: {bs.look_ahead}")
    print(f"  use_bbd: {bs.use_bbd}")
    print(f"  bbd_conservative: {bs.bbd_conservative}")
    
    print("\nScorer Weights:")
    for name, weight in bs.weights.items():
        print(f"  {name}: {weight}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Compare key parameters
espnet_block_size = config.get('encoder_conf', {}).get('block_size', None)
espnet_hop_size = config.get('encoder_conf', {}).get('hop_size', None)
espnet_look_ahead = config.get('encoder_conf', {}).get('look_ahead', None)

print("\nBlock Processing:")
print(f"  ESPnet block_size: {espnet_block_size}")
print(f"  Ours block_size:   {bs.block_size}")
print(f"  Match: {'✅' if espnet_block_size == bs.block_size else '❌'}")

print(f"\n  ESPnet hop_size: {espnet_hop_size}")
print(f"  Ours hop_size:   {bs.hop_size}")
print(f"  Match: {'✅' if espnet_hop_size == bs.hop_size else '❌'}")

print(f"\n  ESPnet look_ahead: {espnet_look_ahead}")
print(f"  Ours look_ahead:   {bs.look_ahead}")
print(f"  Match: {'✅' if espnet_look_ahead == bs.look_ahead else '❌'}")

espnet_ctc_weight = config.get('ctc_weight', 0.0)
our_ctc_weight = bs.weights.get('ctc', 0.0)

print(f"\nCTC Weight:")
print(f"  ESPnet: {espnet_ctc_weight}")
print(f"  Ours:   {our_ctc_weight}")
print(f"  Match: {'✅' if abs(espnet_ctc_weight - our_ctc_weight) < 0.001 else '❌'}")

print("\n" + "="*80)
