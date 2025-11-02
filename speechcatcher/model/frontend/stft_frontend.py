"""STFT-based frontend for feature extraction.

Matches ESPnet's DefaultFrontend pipeline exactly:
STFT → Power Spectrum → Mel Filterbank → Log
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchaudio


class STFTFrontend(nn.Module):
    """STFT-based frontend for extracting log-mel filterbank features.

    This module matches ESPnet's DefaultFrontend pipeline:
    1. STFT (returns complex representation)
    2. Power spectrum: real^2 + imag^2
    3. Mel filterbank transformation (matrix multiplication)
    4. Log transformation

    Args:
        n_fft: FFT size (default: 512)
        hop_length: Hop length between frames (default: 160)
        win_length: Window length (default: 400)
        n_mels: Number of mel filterbanks (default: 80)
        sample_rate: Audio sample rate (default: 16000)
        f_min: Minimum frequency for mel filterbanks (default: 0.0)
        f_max: Maximum frequency for mel filterbanks (default: None, uses sample_rate/2)
        center: Whether to pad waveform on both sides (default: True, matches ESPnet)
        normalized: Whether to normalize STFT (default: False, matches ESPnet)
        onesided: Whether to return one-sided STFT (default: True, matches ESPnet)

    Shape:
        - Input: (batch, samples)
        - Output: (batch, frames, n_mels)
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,  # ESPnet default
        win_length: int = 400,  # ESPnet default
        n_mels: int = 80,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float = None,
        center: bool = True,  # ESPnet default
        normalized: bool = False,  # ESPnet default
        onesided: bool = True,  # ESPnet default
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

        # Create window (Hann window, matches ESPnet default)
        # Register as buffer so it moves to device with model
        self.register_buffer('window', torch.hann_window(win_length))

        # Create mel filterbank matrix using torchaudio
        # torchaudio.functional.melscale_fbanks returns shape (n_freqs, n_mels)
        n_freqs = n_fft // 2 + 1 if onesided else n_fft
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm='slaney',  # ESPnet default (Slaney normalization)
            mel_scale='slaney',  # ESPnet default (htk=False means slaney scale)
        )

        # Register as buffer: (n_freqs, n_mels)
        # This matches ESPnet's melmat which is (n_freqs, n_mels)
        self.register_buffer('mel_fb', mel_fb)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract log-mel filterbank features.

        Follows ESPnet's DefaultFrontend pipeline exactly:
        1. STFT
        2. Power spectrum (real^2 + imag^2)
        3. Mel filterbank transformation
        4. Log

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

        # Step 1: STFT
        # torch.stft returns complex tensor (batch, freq, frames)
        stft_complex = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        # stft_complex: (batch, n_freqs, frames)

        # Transpose to (batch, frames, n_freqs) to match ESPnet
        stft_complex = stft_complex.transpose(1, 2)
        # stft_complex: (batch, frames, n_freqs)

        # Step 2: Power spectrum
        # ESPnet: input_power = input_stft.real**2 + input_stft.imag**2
        power_spec = stft_complex.real ** 2 + stft_complex.imag ** 2
        # power_spec: (batch, frames, n_freqs)

        # Step 3: Mel filterbank transformation
        # ESPnet: mel_feat = torch.matmul(feat, self.melmat)
        # feat: (B, T, n_freqs) x melmat: (n_freqs, n_mels) -> mel_feat: (B, T, n_mels)
        mel_spec = torch.matmul(power_spec, self.mel_fb)
        # mel_spec: (batch, frames, n_mels)

        # Step 4: Log transformation
        # ESPnet: mel_feat = torch.clamp(mel_feat, min=1e-10); logmel_feat = mel_feat.log()
        mel_spec = torch.clamp(mel_spec, min=1e-10)
        log_mel = mel_spec.log()
        # log_mel: (batch, frames, n_mels)

        # Compute lengths (number of frames for each utterance)
        lengths = torch.full(
            (batch_size,),
            log_mel.size(1),
            dtype=torch.long,
            device=log_mel.device,
        )

        return log_mel, lengths

    def output_size(self) -> int:
        """Return the output feature dimension."""
        return self.n_mels
