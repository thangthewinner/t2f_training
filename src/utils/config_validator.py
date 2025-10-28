"""Configuration validation utilities."""

import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    errors = []

    # Check required top-level keys
    required_keys = ['experiment', 'dataset', 'model', 'training']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config section: '{key}'")

    if errors:
        raise ConfigValidationError("\n".join(errors))

    # Validate experiment section
    errors.extend(_validate_experiment(config.get('experiment', {})))

    # Validate dataset section
    errors.extend(_validate_dataset(config.get('dataset', {})))

    # Validate model section
    errors.extend(_validate_model(config.get('model', {})))

    # Validate training section
    errors.extend(_validate_training(config.get('training', {})))

    if errors:
        raise ConfigValidationError("Configuration validation failed:\n" + "\n".join(errors))

    logger.info("Configuration validation passed")


def _validate_experiment(config: Dict[str, Any]) -> List[str]:
    """Validate experiment configuration."""
    errors = []

    # Check device
    device = config.get('device', 'cuda')
    if device not in ['cuda', 'cpu']:
        errors.append(f"Invalid device: '{device}'. Must be 'cuda' or 'cpu'")

    # Check seed
    seed = config.get('seed')
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        errors.append(f"Invalid seed: {seed}. Must be non-negative integer")

    return errors


def _validate_dataset(config: Dict[str, Any]) -> List[str]:
    """Validate dataset configuration."""
    errors = []

    # Check data_root exists
    data_root = config.get('data_root')
    if data_root is None:
        errors.append("Missing 'data_root' in dataset config")
    else:
        data_root_path = Path(data_root)
        if not data_root_path.exists():
            errors.append(f"Data root directory not found: {data_root}")
        else:
            # Check caps_file exists
            caps_file = config.get('caps_file', 'caps.txt')
            caps_path = data_root_path / caps_file
            if not caps_path.exists():
                errors.append(f"Captions file not found: {caps_path}")

            # Check img_dir exists
            img_dir = config.get('img_dir', 'img')
            img_path = data_root_path / img_dir
            if not img_path.exists():
                errors.append(f"Image directory not found: {img_path}")

    # Check train_split
    train_split = config.get('train_split', 0.8)
    if not (0.0 < train_split < 1.0):
        errors.append(f"Invalid train_split: {train_split}. Must be between 0 and 1")

    # Check image_size
    image_size = config.get('image_size', 1024)
    if image_size not in [256, 512, 1024]:
        logger.warning(f"Unusual image_size: {image_size}. Common values are 256, 512, 1024")

    return errors


def _validate_model(config: Dict[str, Any]) -> List[str]:
    """Validate model configuration."""
    errors = []

    # Check latent_space
    latent_space = config.get('latent_space', 'wplus')
    if latent_space not in ['z', 'w', 'wplus']:
        errors.append(f"Invalid latent_space: '{latent_space}'. Must be 'z', 'w', or 'wplus'")

    # Validate text_encoder
    text_encoder_cfg = config.get('text_encoder', {})
    pooling = text_encoder_cfg.get('pooling', 'cls')
    if pooling not in ['cls', 'mean', 'max', 'concat']:
        errors.append(f"Invalid pooling: '{pooling}'. Must be 'cls', 'mean', 'max', or 'concat'")

    # Validate mapper
    mapper_cfg = config.get('mapper', {})
    output_dim = mapper_cfg.get('output_dim')
    if output_dim is None:
        errors.append("Missing 'output_dim' in mapper config")
    elif output_dim != 512:
        logger.warning(f"output_dim is {output_dim}, but should typically be 512 for StyleGAN2")

    intermediate_dims = mapper_cfg.get('intermediate_dims', [])
    if not isinstance(intermediate_dims, list) or len(intermediate_dims) == 0:
        errors.append("'intermediate_dims' must be a non-empty list")

    # Validate StyleGAN2
    stylegan_cfg = config.get('stylegan2', {})
    model_path = stylegan_cfg.get('model_path')
    if model_path is None:
        errors.append("Missing 'model_path' in stylegan2 config")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            errors.append(f"StyleGAN2 checkpoint not found: {model_path}")

    truncation_psi = stylegan_cfg.get('truncation_psi', 0.7)
    if not (0.0 <= truncation_psi <= 1.0):
        errors.append(f"Invalid truncation_psi: {truncation_psi}. Must be between 0 and 1")

    # Validate loss
    loss_cfg = config.get('loss', {})
    content_weight = loss_cfg.get('content_weight', 1.0)
    style_weight = loss_cfg.get('style_weight', 0.0)
    tv_weight = loss_cfg.get('tv_weight', 1e-6)

    # Convert to float in case they're strings (e.g., "1e-6" from YAML)
    try:
        content_weight = float(content_weight)
        style_weight = float(style_weight)
        tv_weight = float(tv_weight)
    except (ValueError, TypeError):
        errors.append("Loss weights must be numeric values")
        return errors

    if content_weight < 0 or style_weight < 0 or tv_weight < 0:
        errors.append("Loss weights must be non-negative")

    if content_weight == 0 and style_weight == 0:
        logger.warning("Both content_weight and style_weight are 0. Model may not train properly")

    return errors


def _validate_training(config: Dict[str, Any]) -> List[str]:
    """Validate training configuration."""
    errors = []

    # Check epochs
    epochs = config.get('epochs')
    if epochs is None or epochs <= 0:
        errors.append(f"Invalid epochs: {epochs}. Must be positive integer")

    # Check batch_size
    batch_size = config.get('batch_size', 1)
    if batch_size <= 0:
        errors.append(f"Invalid batch_size: {batch_size}. Must be positive")

    # Check gradient_accumulation_steps
    grad_accum = config.get('gradient_accumulation_steps', 1)
    if grad_accum <= 0:
        errors.append(f"Invalid gradient_accumulation_steps: {grad_accum}. Must be positive")

    # Check learning rate
    optimizer_cfg = config.get('optimizer', {})
    lr = optimizer_cfg.get('lr')
    if lr is None:
        errors.append("Missing 'lr' in optimizer config")
    elif lr <= 0 or lr > 0.1:
        logger.warning(f"Unusual learning rate: {lr}. Typical range is 1e-5 to 1e-2")

    # Check scheduler
    scheduler_cfg = optimizer_cfg.get('scheduler', {})
    if scheduler_cfg:
        sched_type = scheduler_cfg.get('type', 'cosine')
        valid_types = ['cosine', 'step', 'plateau', 'none']
        if sched_type not in valid_types:
            errors.append(f"Invalid scheduler type: '{sched_type}'. Must be one of {valid_types}")

    return errors
