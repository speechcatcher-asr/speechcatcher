"""STFT-based frontend for feature extraction."""

from typing import Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as T


class STFTFrontend(nn.Module):
    """STFT-based frontend for extracting log-mel filterbank features.

    This module converts raw audio waveforms to log-mel spectrograms using STFT.
    Compatible with ESPnet's default frontend configuration.

    Args:
        n_fft: FFT size (default: 512)
        hop_length: Hop length between frames (default: 128)
        win_length: Window length (default: 512, same as n_fft)
        n_mels: Number of mel filterbanks (default: 80)
        sample_rate: Audio sample rate (default: 16000)
        f_min: Minimum frequency for mel filterbanks (default: 0.0)
        f_max: Maximum frequency for mel filterbanks (default: None, uses sample_rate/2)
        window_fn: Window function (default: torch.hann_window)

    Shape:
        - Input: (batch, samples)
        - Output: (batch, frames, n_mels)
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_mels: int = 80,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float = None,
        window_fn=torch.hann_window,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # Create mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2.0,
            n_mels=n_mels,
            window_fn=window_fn,
            power=2.0,  # Power spectrum (squared magnitude)
        )

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract log-mel filterbank features.

        Args:
            waveform: Input waveform (batch, samples)

        Returns:
            Tuple of:
                - features: Log-mel features (batch, frames, n_mels)
                - lengths: Feature lengths (batch,)
        """
        # Ensure waveform is 2D (batch, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.size(0)
        n_samples = waveform.size(1)

        # Compute mel spectrogram: (batch, n_mels, frames)
        mel_spec = self.mel_spectrogram(waveform)

        # Apply log (add small epsilon for numerical stability)
        log_mel = torch.log(mel_spec + 1e-10)

        # Transpose to (batch, frames, n_mels)
        features = log_mel.transpose(1, 2)

        # Compute lengths (number of frames for each utterance)
        # frames = (samples - win_length) // hop_length + 1
        # But MelSpectrogram handles this internally, so we get the actual size
        lengths = torch.full(
            (batch_size,),
            features.size(1),
            dtype=torch.long,
            device=features.device,
        )

        return features, lengths

    def output_size(self) -> int:
        """Return the output feature dimension."""
        return self.n_mels
