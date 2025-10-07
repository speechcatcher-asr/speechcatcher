"""Model modules for speechcatcher."""

from speechcatcher.model.ctc import CTC
from speechcatcher.model.espnet_asr_model import ESPnetASRModel

__all__ = [
    "CTC",
    "ESPnetASRModel",
]
