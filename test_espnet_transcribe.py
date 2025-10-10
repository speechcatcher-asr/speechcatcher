#!/usr/bin/env python3
"""Test what ESPnet actually transcribes for our sample audio."""

import logging
import torch
import numpy as np
import wave
import hashlib
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("ESPnet Full Transcription Test")
logger.info("="*80)

# Load audio
from speechcatcher.speechcatcher import convert_inputfile

os.makedirs('.tmp/', exist_ok=True)
wavfile_path = '.tmp/' + hashlib.sha1("Neujahrsansprache_5s.mp4".encode("utf-8")).hexdigest() + '.wav'
if not os.path.exists(wavfile_path):
    convert_inputfile("Neujahrsansprache_5s.mp4", wavfile_path)

with wave.open(wavfile_path, 'rb') as wavfile_in:
    buf = wavfile_in.readframes(-1)
    raw_audio = np.frombuffer(buf, dtype='int16')

waveform = raw_audio.astype(np.float32)
logger.info(f"Audio loaded: {waveform.shape}")

# Load ESPnet model
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming as ESPnetS2T

config_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/config.yaml"
model_path = "/home/ben/.cache/espnet/models--speechcatcher--speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024/snapshots/469c3474c28025d77cd7c1e1671638b56de53c2d/exp/asr_train_asr_streaming_transformer_size_xl_raw_de_bpe1024/valid.acc.ave_6best.pth"

espnet_model = ESPnetS2T(
    asr_train_config=config_path,
    asr_model_file=model_path,
    device="cpu",
)

logger.info("✅ ESPnet model loaded")

# Transcribe
logger.info("\n" + "="*80)
logger.info("Running ESPnet transcription...")
logger.info("="*80)

results = espnet_model(waveform)

logger.info(f"\nESPnet transcription result:")
logger.info(f"Result type: {type(results)}")
if isinstance(results, list) and len(results) > 0:
    for i, result in enumerate(results):
        logger.info(f"\nResult {i+1}:")
        logger.info(f"  Type: {type(result)}")
        logger.info(f"  Length: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}")

        if isinstance(result, (list, tuple)):
            logger.info(f"  Contents: {result}")
            # Try to extract text and token_ids
            if len(result) >= 2:
                text = result[0] if isinstance(result[0], str) else str(result[0])
                token_ids = result[1] if isinstance(result[1], list) else []
                logger.info(f"  Text: '{text}'")
                if token_ids:
                    logger.info(f"  Token IDs ({len(token_ids)}): {token_ids[:20]}...")

                    # Check if token 1023 appears
                    if 1023 in token_ids:
                        count = token_ids.count(1023)
                        positions = [j for j, x in enumerate(token_ids) if x == 1023]
                        logger.warning(f"  ⚠️  Token 1023 appears {count} times at positions: {positions[:10]}")
                    else:
                        logger.info(f"  ✅ Token 1023 does NOT appear in output")
        else:
            logger.info(f"  {result}")
else:
    logger.info(f"  {results}")

logger.info("\n" + "="*80)
logger.info("ESPnet Transcription Test Complete")
logger.info("="*80)
