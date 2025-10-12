#!/usr/bin/env python3
"""Check if models are in eval mode and test dropout behavior."""

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Load both models
# ============================================================================
logger.info("="*80)
logger.info("MODEL MODE CHECK")
logger.info("="*80)

from speechcatcher.speechcatcher import load_model
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming as ESPnetS2T

logger.info("\n[1] Loading models...")

our_model = load_model(
    tag="speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    device="cpu",
    beam_size=5,
    quiet=True
)

espnet_model = ESPnetS2T(
    asr_train_config="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml",
    asr_model_file="/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth",
    device="cpu",
)

logger.info("✅ Models loaded")

# ============================================================================
# Check training mode
# ============================================================================
logger.info("\n[2] Checking training mode...")

logger.info(f"\nOur model training: {our_model.model.training}")
logger.info(f"ESPnet model training: {espnet_model.asr_model.training}")

# Check decoder specifically
logger.info(f"\nOur decoder training: {our_model.model.decoder.training}")
logger.info(f"ESPnet decoder training: {espnet_model.asr_model.decoder.training}")

# Check for dropout layers
logger.info("\n[3] Checking dropout layers...")

def check_dropout_layers(model, name):
    dropout_layers = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_layers.append((module_name, module.p, module.training))

    logger.info(f"\n{name} dropout layers:")
    if dropout_layers:
        for name, p, training in dropout_layers[:10]:  # Show first 10
            logger.info(f"  {name}: p={p:.2f}, training={training}")
        if len(dropout_layers) > 10:
            logger.info(f"  ... and {len(dropout_layers) - 10} more")
    else:
        logger.info("  No dropout layers found")

    return dropout_layers

our_dropouts = check_dropout_layers(our_model.model, "Our model")
espnet_dropouts = check_dropout_layers(espnet_model.asr_model, "ESPnet model")

# ============================================================================
# Set both to eval mode explicitly
# ============================================================================
logger.info("\n[4] Setting both to eval mode...")

our_model.model.eval()
espnet_model.asr_model.eval()

logger.info(f"Our model training: {our_model.model.training}")
logger.info(f"ESPnet model training: {espnet_model.asr_model.training}")

# ============================================================================
# Test with same random input to verify determinism
# ============================================================================
logger.info("\n[5] Testing determinism with same random input...")

torch.manual_seed(42)
input1 = torch.randn(1, 124, 256)

torch.manual_seed(42)
input2 = torch.randn(1, 124, 256)

logger.info(f"Inputs identical: {torch.allclose(input1, input2)}")

# Test our decoder
sos_id = 1
yseq = torch.tensor([sos_id])

with torch.no_grad():
    our_out1, _ = our_model.model.decoder.score(yseq, state=None, x=input1[0])
    our_out2, _ = our_model.model.decoder.score(yseq, state=None, x=input2[0])

logger.info(f"\nOur decoder outputs identical: {torch.allclose(our_out1, our_out2)}")
logger.info(f"  Max diff: {(our_out1 - our_out2).abs().max():.6f}")

# Test ESPnet decoder
with torch.no_grad():
    espnet_out1, _ = espnet_model.asr_model.decoder.score(yseq, state=None, x=input1[0])
    espnet_out2, _ = espnet_model.asr_model.decoder.score(yseq, state=None, x=input2[0])

logger.info(f"\nESPnet decoder outputs identical: {torch.allclose(espnet_out1, espnet_out2)}")
logger.info(f"  Max diff: {(espnet_out1 - espnet_out2).abs().max():.6f}")

logger.info("\n" + "="*80)
logger.info("MODEL MODE CHECK COMPLETE")
logger.info("="*80)
