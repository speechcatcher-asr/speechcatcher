#!/usr/bin/env python3
"""Comprehensive decoder weight comparison to identify loading issues."""

import logging
import sys
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: Load checkpoint and extract all decoder keys
# ============================================================================
logger.info("="*80)
logger.info("STEP 1: ANALYZING CHECKPOINT STRUCTURE")
logger.info("="*80)

ckpt_path = Path('/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth')

logger.info(f"Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Get all decoder keys from checkpoint
decoder_keys = sorted([k for k in checkpoint.keys() if k.startswith('decoder.')])

logger.info(f"\nFound {len(decoder_keys)} decoder parameters in checkpoint")
logger.info("\nDecoder parameter categories:")

# Categorize keys
categories = {
    'embed': [],
    'decoders': [],
    'after_norm': [],
    'output_layer': [],
    'other': []
}

for key in decoder_keys:
    if 'embed' in key:
        categories['embed'].append(key)
    elif 'decoders' in key:
        categories['decoders'].append(key)
    elif 'after_norm' in key:
        categories['after_norm'].append(key)
    elif 'output_layer' in key:
        categories['output_layer'].append(key)
    else:
        categories['other'].append(key)

for cat_name, keys in categories.items():
    if keys:
        logger.info(f"\n  {cat_name}: {len(keys)} parameters")
        for key in keys[:3]:  # Show first 3
            value = checkpoint[key]
            if hasattr(value, 'shape'):
                logger.info(f"    {key}: {value.shape}")
            else:
                logger.info(f"    {key}: {type(value)}")
        if len(keys) > 3:
            logger.info(f"    ... and {len(keys)-3} more")

# ============================================================================
# STEP 2: Load our model and extract all decoder parameters
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 2: ANALYZING OUR MODEL STRUCTURE")
logger.info("="*80)

from speechcatcher.speechcatcher import load_model

speech2text = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)

# Get all decoder parameters from our model
model_params = {}
for name, param in speech2text.model.named_parameters():
    if name.startswith('decoder.'):
        model_params[name] = param

logger.info(f"\nFound {len(model_params)} decoder parameters in our model")

# Categorize our model params
model_categories = {
    'embed': [],
    'decoders': [],
    'after_norm': [],
    'output_layer': [],
    'other': []
}

for key in sorted(model_params.keys()):
    if 'embed' in key:
        model_categories['embed'].append(key)
    elif 'decoders' in key:
        model_categories['decoders'].append(key)
    elif 'after_norm' in key:
        model_categories['after_norm'].append(key)
    elif 'output_layer' in key:
        model_categories['output_layer'].append(key)
    else:
        model_categories['other'].append(key)

logger.info("\nOur model parameter categories:")
for cat_name, keys in model_categories.items():
    if keys:
        logger.info(f"\n  {cat_name}: {len(keys)} parameters")
        for key in keys[:3]:  # Show first 3
            param = model_params[key]
            logger.info(f"    {key}: {param.shape}")
        if len(keys) > 3:
            logger.info(f"    ... and {len(keys)-3} more")

# ============================================================================
# STEP 3: Compare checkpoint vs model - find missing weights
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 3: COMPARING CHECKPOINT VS MODEL")
logger.info("="*80)

# Find which checkpoint keys are NOT in our model
missing_in_model = set(decoder_keys) - set(model_params.keys())

# Find which model params are NOT in checkpoint
missing_in_checkpoint = set(model_params.keys()) - set(decoder_keys)

# Find common keys
common_keys = set(decoder_keys) & set(model_params.keys())

logger.info(f"\nComparison summary:")
logger.info(f"  Checkpoint has: {len(decoder_keys)} parameters")
logger.info(f"  Our model has: {len(model_params)} parameters")
logger.info(f"  Common: {len(common_keys)} parameters")
logger.info(f"  Missing in model: {len(missing_in_model)} parameters")
logger.info(f"  Missing in checkpoint: {len(missing_in_checkpoint)} parameters")

if missing_in_model:
    logger.warning(f"\n❌ {len(missing_in_model)} parameters in checkpoint but NOT in our model:")
    for key in sorted(missing_in_model)[:20]:  # Show first 20
        value = checkpoint[key]
        if hasattr(value, 'shape'):
            logger.warning(f"  {key}: {value.shape}")
    if len(missing_in_model) > 20:
        logger.warning(f"  ... and {len(missing_in_model)-20} more")

if missing_in_checkpoint:
    logger.warning(f"\n❌ {len(missing_in_checkpoint)} parameters in our model but NOT in checkpoint:")
    for key in sorted(missing_in_checkpoint)[:20]:
        param = model_params[key]
        logger.warning(f"  {key}: {param.shape}")
    if len(missing_in_checkpoint) > 20:
        logger.warning(f"  ... and {len(missing_in_checkpoint)-20} more")

# ============================================================================
# STEP 4: For common keys, compare actual values
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 4: COMPARING VALUES FOR COMMON PARAMETERS")
logger.info("="*80)

mismatched_weights = []
matched_weights = []

logger.info(f"\nChecking {len(common_keys)} common parameters...")

for key in sorted(common_keys):
    ckpt_weight = checkpoint[key]
    model_weight = model_params[key].data  # Get tensor data

    # Compare
    if torch.allclose(ckpt_weight, model_weight, atol=1e-6):
        matched_weights.append(key)
    else:
        mismatched_weights.append(key)
        # Calculate difference
        diff = (ckpt_weight - model_weight).abs()
        logger.error(f"  ❌ MISMATCH: {key}")
        logger.error(f"     Checkpoint: mean={ckpt_weight.mean():.6f}, std={ckpt_weight.std():.6f}")
        logger.error(f"     Our model:  mean={model_weight.mean():.6f}, std={model_weight.std():.6f}")
        logger.error(f"     Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

logger.info(f"\n✅ Matched: {len(matched_weights)} parameters")
if mismatched_weights:
    logger.error(f"❌ Mismatched: {len(mismatched_weights)} parameters")
else:
    logger.info("✅ All common parameters match!")

# ============================================================================
# STEP 5: Special focus on critical weights
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 5: CHECKING CRITICAL DECODER COMPONENTS")
logger.info("="*80)

critical_checks = {
    'Embedding': 'decoder.embed.0.weight',
    'Output Layer Weight': 'decoder.output_layer.weight',
    'Output Layer Bias': 'decoder.output_layer.bias',
    'After Norm Weight': 'decoder.after_norm.weight',
    'After Norm Bias': 'decoder.after_norm.bias',
    'First Decoder Layer - Self Attention Q': 'decoder.decoders.0.self_attn.linear_q.weight',
    'First Decoder Layer - Self Attention K': 'decoder.decoders.0.self_attn.linear_k.weight',
    'First Decoder Layer - Self Attention V': 'decoder.decoders.0.self_attn.linear_v.weight',
    'First Decoder Layer - Cross Attention Q': 'decoder.decoders.0.src_attn.linear_q.weight',
}

for desc, key in critical_checks.items():
    logger.info(f"\n{desc}: {key}")

    in_checkpoint = key in decoder_keys
    in_model = key in model_params

    logger.info(f"  In checkpoint: {in_checkpoint}")
    logger.info(f"  In our model: {in_model}")

    if in_checkpoint and in_model:
        ckpt_weight = checkpoint[key]
        model_weight = model_params[key].data

        if torch.allclose(ckpt_weight, model_weight, atol=1e-6):
            logger.info(f"  ✅ VALUES MATCH")
        else:
            logger.error(f"  ❌ VALUES DIFFER!")
            diff = (ckpt_weight - model_weight).abs()
            logger.error(f"     Max diff: {diff.max():.6f}")
    elif in_checkpoint and not in_model:
        logger.error(f"  ❌ EXISTS IN CHECKPOINT BUT NOT LOADED!")
    elif not in_checkpoint and in_model:
        logger.warning(f"  ⚠️  EXISTS IN MODEL BUT NOT IN CHECKPOINT (random init?)")

# ============================================================================
# STEP 6: Test decoder output_layer specifically
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 6: TESTING OUTPUT_LAYER (FINAL PROJECTION)")
logger.info("="*80)

# The output_layer maps hidden states (256-dim) to vocabulary logits (1024-dim)
# This is THE most critical layer for predicting tokens

if 'decoder.output_layer.weight' in decoder_keys and 'decoder.output_layer.weight' in model_params:
    logger.info("\nTesting output_layer with random hidden state...")

    # Create a random hidden state
    hidden = torch.randn(1, 256)

    # Apply checkpoint output_layer
    ckpt_output_weight = checkpoint['decoder.output_layer.weight']
    if 'decoder.output_layer.bias' in checkpoint:
        ckpt_output_bias = checkpoint['decoder.output_layer.bias']
        ckpt_logits = torch.nn.functional.linear(hidden, ckpt_output_weight, ckpt_output_bias)
    else:
        ckpt_logits = torch.nn.functional.linear(hidden, ckpt_output_weight)

    # Apply our model's output_layer
    model_output_layer = speech2text.model.decoder.output_layer
    model_logits = model_output_layer(hidden)

    # Compare
    if torch.allclose(ckpt_logits, model_logits, atol=1e-5):
        logger.info("  ✅ Output layer produces IDENTICAL results!")
    else:
        logger.error("  ❌ Output layer produces DIFFERENT results!")
        diff = (ckpt_logits - model_logits).abs()
        logger.error(f"     Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

        # Check which token has max prediction in each
        ckpt_top_token = ckpt_logits.argmax().item()
        model_top_token = model_logits.argmax().item()
        logger.info(f"     Checkpoint predicts token: {ckpt_top_token}")
        logger.info(f"     Our model predicts token: {model_top_token}")

        if model_top_token == 1023:
            logger.error(f"     ❌ Our model predicts token 1023 (the problematic token)!")

logger.info("\n" + "="*80)
logger.info("WEIGHT COMPARISON COMPLETE")
logger.info("="*80)

# ============================================================================
# SUMMARY
# ============================================================================
logger.info("\n" + "="*80)
logger.info("SUMMARY")
logger.info("="*80)

logger.info(f"\nCheckpoint structure: {len(decoder_keys)} parameters")
logger.info(f"Our model structure: {len(model_params)} parameters")
logger.info(f"Common parameters: {len(common_keys)}")
logger.info(f"Missing in model: {len(missing_in_model)}")
logger.info(f"Missing in checkpoint: {len(missing_in_checkpoint)}")
logger.info(f"Matched values: {len(matched_weights)}")
logger.info(f"Mismatched values: {len(mismatched_weights)}")

if missing_in_model:
    logger.error(f"\n❌ CRITICAL: {len(missing_in_model)} checkpoint parameters are NOT loaded into our model!")
    logger.error("   This is likely THE ROOT CAUSE of wrong predictions.")

if mismatched_weights:
    logger.error(f"\n❌ CRITICAL: {len(mismatched_weights)} parameters have DIFFERENT values!")
    logger.error("   Even though they exist, they don't match the checkpoint.")

if not missing_in_model and not mismatched_weights:
    logger.info("\n✅ All checkpoint parameters are loaded correctly!")
    logger.info("   The issue must be elsewhere (architecture mismatch, etc.)")
