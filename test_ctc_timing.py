#!/usr/bin/env python3
"""Test CTC timing with detailed logging."""

import logging
import sys
import time

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

# Import after logging setup
from speechcatcher.speech2text_streaming import Speech2TextStreaming

logger.info("Loading model...")
# Manually load with higher CTC weight
from espnet_model_zoo.downloader import ModelDownloader
from pathlib import Path
from speechcatcher.speech2text_streaming import Speech2TextStreaming

espnet_model_downloader = ModelDownloader("~/.cache/espnet")
info = espnet_model_downloader.download_and_unpack(
    "speechcatcher/speechcatcher_german_espnet_streaming_transformer_26k_train_size_xl_raw_de_bpe1024",
    quiet=False
)

model_dir = None
for key in ['asr_model_file', 'asr_train_config', 'model_file', 'train_config']:
    if key in info and info[key]:
        model_dir = Path(info[key]).parent
        break

logger.info(f"Model dir: {model_dir}")

speech2text = Speech2TextStreaming(
    model_dir=model_dir,
    beam_size=10,
    ctc_weight=0.0,  # DISABLE CTC to test decoder-only
    device="cpu",
    dtype="float32"
)

logger.info("Model loaded successfully")

# Load audio
import torchaudio
logger.info("Loading audio file...")
waveform, sample_rate = torchaudio.load("Neujahrsansprache_5s.mp4")

# Resample to 16kHz if needed
if sample_rate != 16000:
    logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    sample_rate = 16000

# Convert to mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

logger.info(f"Audio loaded: shape={waveform.shape}, duration={waveform.shape[1]/sample_rate:.2f}s")

# Check beam search configuration
logger.info(f"Beam search scorers: {list(speech2text.beam_search.scorers.keys())}")
logger.info(f"Beam search weights: {speech2text.beam_search.weights}")

# Process audio
start_time = time.time()
logger.info("Starting transcription...")

try:
    results = speech2text(speech=waveform.squeeze(0).numpy(), is_final=True)
    elapsed = time.time() - start_time
    logger.info(f"Transcription complete in {elapsed:.2f}s")

    print("\n=== RESULTS ===")
    print(results)

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    logger.error(f"Interrupted after {elapsed:.2f}s")
    sys.exit(1)
except Exception as e:
    elapsed = time.time() - start_time
    logger.error(f"Error after {elapsed:.2f}s: {e}", exc_info=True)
    sys.exit(1)
