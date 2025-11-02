"""Utilities for loading ESPnet model checkpoints."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """Load ESPnet configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load ESPnet checkpoint file.

    Args:
        checkpoint_path: Path to .pth checkpoint file

    Returns:
        Checkpoint dictionary containing model state_dict and other metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def infer_model_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict:
    """Infer model architecture parameters from state_dict.

    Args:
        state_dict: Model state dictionary

    Returns:
        Dictionary with architecture parameters
    """
    arch = {}

    # Find encoder parameters
    encoder_keys = [k for k in state_dict.keys() if k.startswith("encoder.")]

    if encoder_keys:
        # Count encoder layers
        encoder_layer_indices = set()
        for key in encoder_keys:
            if "encoders." in key:
                parts = key.split(".")
                idx = parts[parts.index("encoders") + 1]
                if idx.isdigit():
                    encoder_layer_indices.add(int(idx))

        arch["num_encoder_layers"] = max(encoder_layer_indices) + 1 if encoder_layer_indices else 0

        # Get model dimension from embedding layer
        embed_key = "encoder.embed.0.weight"  # Conv2d weight
        if embed_key in state_dict:
            arch["encoder_output_size"] = state_dict[embed_key].shape[0]
        else:
            # Try to infer from attention weights
            for key in encoder_keys:
                if "self_attn.linear_q.weight" in key:
                    arch["encoder_output_size"] = state_dict[key].shape[1]
                    break

        # Get attention heads from attention layer
        for key in encoder_keys:
            if "self_attn.linear_q.weight" in key:
                d_model = state_dict[key].shape[1]
                d_k = state_dict[key].shape[0]
                arch["encoder_attention_heads"] = d_k // (d_model // arch.get("encoder_output_size", d_model))
                break

    # Find decoder parameters
    decoder_keys = [k for k in state_dict.keys() if k.startswith("decoder.")]

    if decoder_keys:
        # Count decoder layers
        decoder_layer_indices = set()
        for key in decoder_keys:
            if "decoders." in key:
                parts = key.split(".")
                idx = parts[parts.index("decoders") + 1]
                if idx.isdigit():
                    decoder_layer_indices.add(int(idx))

        arch["num_decoder_layers"] = max(decoder_layer_indices) + 1 if decoder_layer_indices else 0

        # Get vocab size from output layer
        if "decoder.output_layer.weight" in state_dict:
            arch["vocab_size"] = state_dict["decoder.output_layer.weight"].shape[0]
        elif "decoder.embed.0.weight" in state_dict:  # Embedding layer
            arch["vocab_size"] = state_dict["decoder.embed.0.weight"].shape[0]

        # Get attention heads from decoder attention
        for key in decoder_keys:
            if "self_attn.linear_q.weight" in key:
                d_model = state_dict[key].shape[1]
                d_k = state_dict[key].shape[0]
                arch["decoder_attention_heads"] = d_k // (d_model // arch.get("encoder_output_size", d_model))
                break

    # Find CTC parameters
    ctc_keys = [k for k in state_dict.keys() if k.startswith("ctc.")]
    if ctc_keys and "ctc.ctc_lo.weight" in state_dict:
        arch["ctc_vocab_size"] = state_dict["ctc.ctc_lo.weight"].shape[0]

    logger.info(f"Inferred architecture: {arch}")
    return arch


def map_espnet_to_speechcatcher(espnet_key: str) -> Optional[str]:
    """Map ESPnet parameter name to speechcatcher parameter name.

    Args:
        espnet_key: ESPnet parameter name

    Returns:
        Corresponding speechcatcher parameter name, or None if not mappable
    """
    # Most keys map directly - ESPnet and speechcatcher use same structure
    # Just return the key as-is for most cases
    if espnet_key.startswith("encoder."):
        return espnet_key

    if espnet_key.startswith("decoder."):
        return espnet_key

    if espnet_key.startswith("ctc."):
        return espnet_key

    # Frontend and normalization stats - these might not exist in our model
    if espnet_key.startswith("frontend.") or espnet_key.startswith("normalize_"):
        return None  # Skip frontend parameters

    return espnet_key  # Pass through other keys


def load_espnet_weights(
    model: nn.Module,
    checkpoint_path: Path,
    strict: bool = False,
) -> Tuple[nn.Module, Dict]:
    """Load ESPnet checkpoint weights into speechcatcher model.

    Args:
        model: Speechcatcher model (Encoder, Decoder, or full ASR model)
        checkpoint_path: Path to ESPnet .pth checkpoint
        strict: Whether to strictly enforce weight loading (default: False)

    Returns:
        Tuple of (model with loaded weights, architecture info dict)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Extract state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Infer architecture
    arch_info = infer_model_architecture(state_dict)

    # Map parameter names
    mapped_state_dict = {}
    unmapped_keys = []

    for key, value in state_dict.items():
        mapped_key = map_espnet_to_speechcatcher(key)
        if mapped_key:
            mapped_state_dict[mapped_key] = value
        else:
            unmapped_keys.append(key)

    if unmapped_keys:
        logger.debug(f"Unmapped keys ({len(unmapped_keys)}): {unmapped_keys[:10]}...")

    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=strict)

    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}...")

    logger.info(f"Successfully loaded {len(mapped_state_dict)} parameters")

    return model, arch_info


def load_normalization_stats(stats_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature normalization statistics.

    Args:
        stats_path: Path to feats_stats.npz

    Returns:
        Tuple of (mean, std) arrays
    """
    stats = np.load(stats_path)

    # Check if stored as mean/std or sum/sum_square/count
    if "mean" in stats:
        mean = stats["mean"]
        std = stats["std"]
    elif "sum" in stats and "sum_square" in stats and "count" in stats:
        # Compute mean and std from accumulated statistics
        count = stats["count"]
        mean = stats["sum"] / count
        # std = sqrt(E[X^2] - E[X]^2)
        mean_square = stats["sum_square"] / count
        std = np.sqrt(np.maximum(mean_square - mean ** 2, 1e-10))
    else:
        raise ValueError(f"Unknown stats format. Keys: {list(stats.keys())}")

    logger.info(f"Loaded normalization stats: mean shape {mean.shape}, std shape {std.shape}")

    return mean, std


def apply_feature_normalization(
    features: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Apply mean-variance normalization to features.

    Args:
        features: Input features (batch, time, feat_dim) or (time, feat_dim)
        mean: Mean array (feat_dim,)
        std: Standard deviation array (feat_dim,)

    Returns:
        Normalized features (same shape as input)
    """
    mean_tensor = torch.from_numpy(mean).to(features.device).to(features.dtype)
    std_tensor = torch.from_numpy(std).to(features.device).to(features.dtype)

    # Normalize
    features = (features - mean_tensor) / std_tensor

    return features


def load_espnet_model_from_directory(
    model_dir: Path,
    model: nn.Module,
    checkpoint_name: str = "valid.acc.best.pth",
    stats_name: str = "feats_stats.npz",
    config_name: str = "config.yaml",
) -> Tuple[nn.Module, Dict, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Load ESPnet model from a directory containing checkpoint, config, and stats.

    Args:
        model_dir: Directory containing model files
        model: Speechcatcher model to load weights into
        checkpoint_name: Name of checkpoint file
        stats_name: Name of stats file
        config_name: Name of config file

    Returns:
        Tuple of (model with loaded weights, config dict, (mean, std) or None)
    """
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / config_name
    if config_path.exists():
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}")
        config = {}

    # Load checkpoint
    checkpoint_path = model_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, arch_info = load_espnet_weights(model, checkpoint_path, strict=False)

    # Load normalization stats
    stats_path = model_dir / stats_name
    if stats_path.exists():
        mean, std = load_normalization_stats(stats_path)
        stats = (mean, std)
        logger.info(f"Loaded normalization stats from {stats_path}")
    else:
        logger.warning(f"Stats file not found: {stats_path}")
        stats = None

    # Merge architecture info into config
    config["inferred_architecture"] = arch_info

    return model, config, stats
