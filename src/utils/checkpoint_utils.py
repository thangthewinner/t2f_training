"""Checkpoint utilities for resume training and inference."""

import torch
import yaml
import json
import logging
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_config_and_create_experiment_dir(
    checkpoint_path: Path,
    validate_config: bool = True
) -> Tuple[Dict[str, Any], Path]:
    """
    Extract config from checkpoint and create new experiment directory.
    
    This is the primary function for standalone checkpoint resume.
    Ported from training_final.ipynb with enhanced validation.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        validate_config: Whether to validate config structure
        
    Returns:
        (config, new_experiment_dir)
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint missing required keys
        
    Example:
        >>> ckpt_path = Path("checkpoint_epoch0050.pt")
        >>> config, exp_dir = extract_config_and_create_experiment_dir(ckpt_path)
        >>> print(exp_dir)
        experiments/resumed_t2f_wplus_from_epoch0050_20241028_153045
    """
    # 1. Validate checkpoint path exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            f"Please check the path and try again."
        )
    
    # 2. Load checkpoint (CPU to avoid GPU memory issues)
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise ValueError(
            f"Failed to load checkpoint: {e}\n"
            f"The checkpoint file may be corrupted."
        )
    
    # 3. Validate checkpoint structure (LENIENT)
    warnings = validate_checkpoint_structure(checkpoint, mode='resume')
    for warning in warnings:
        logger.warning(warning)
    
    # 4. Extract config
    if 'config' not in checkpoint:
        raise ValueError(
            f"Checkpoint missing 'config' key.\n\n"
            f"This checkpoint may be from an older version.\n"
            f"To resume training, please provide config file manually:\n"
            f"  python scripts/train.py --config your_config.yaml --resume {checkpoint_path}\n"
        )
    
    config = checkpoint['config']
    epoch = checkpoint.get('epoch', 0)
    
    # 5. Validate config structure (if requested)
    if validate_config:
        try:
            from .config_validator import validate_config as validate_config_fn
            validate_config_fn(config)
            logger.info("✅ Config validation passed")
        except ImportError:
            logger.warning("Config validator not available, skipping validation")
        except Exception as e:
            logger.warning(f"Config validation warning: {e}")
    
    # 6. Create new experiment directory
    exp_name = config.get('experiment', {}).get('name', 't2f_experiment')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_experiment_dir = Path("experiments") / f"resumed_{exp_name}_from_epoch{epoch:04d}_{timestamp}"
    new_experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ Extracted config from checkpoint (epoch {epoch})")
    logger.info(f"🆕 Created new experiment directory: {new_experiment_dir}")
    
    # 7. Save extracted config to new directory
    config_save_path = new_experiment_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    logger.info(f"✅ Saved extracted config to: {config_save_path}")
    
    # 8. Save resume metadata
    resume_info = {
        'resumed_from': str(checkpoint_path.absolute()),
        'resumed_at': timestamp,
        'resumed_epoch': epoch,
        'resumed_step': checkpoint.get('step', 0),
    }
    resume_info_path = new_experiment_dir / "resume_info.json"
    with open(resume_info_path, "w") as f:
        json.dump(resume_info, f, indent=2)
    logger.info(f"✅ Saved resume info to: {resume_info_path}")
    
    return config, new_experiment_dir


def validate_checkpoint_structure(
    checkpoint: Dict[str, Any],
    mode: str = 'resume'
) -> List[str]:
    """
    Validate checkpoint structure and return warnings (LENIENT approach).
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        mode: 'resume' (training) or 'inference' (generation)
        
    Returns:
        List of warning messages (empty if all good)
        
    Raises:
        ValueError: Only for critical missing keys
    """
    warnings = []
    
    # Critical keys (MUST have)
    if mode == 'resume':
        critical_keys = ['model_state_dict', 'config']
        optional_keys = ['optimizer_state_dict', 'scheduler_state_dict', 
                        'history', 'epoch', 'step']
    else:  # inference
        critical_keys = ['model_state_dict', 'config']
        optional_keys = []
    
    # Check critical keys
    missing_critical = [k for k in critical_keys if k not in checkpoint]
    if missing_critical:
        raise ValueError(
            f"Checkpoint missing critical keys: {missing_critical}\n"
            f"This checkpoint may be corrupted or incompatible."
        )
    
    # Check optional keys (just warn)
    missing_optional = [k for k in optional_keys if k not in checkpoint]
    if missing_optional:
        warnings.append(
            f"Checkpoint missing optional keys: {missing_optional}. "
            f"Training may start from default state for these components."
        )
    
    # Check epoch/step consistency
    if 'epoch' in checkpoint and 'step' not in checkpoint:
        warnings.append("Checkpoint has 'epoch' but not 'step'. Step will default to 0.")
    
    return warnings


def load_config_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load config from checkpoint (lightweight, no experiment dir creation).
    
    Used for inference where we don't need full experiment setup.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Config dictionary
        
    Example:
        >>> config = load_config_from_checkpoint(Path("model.pt"))
        >>> model = T2FModel(config=config)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")
    
    if 'config' not in checkpoint:
        raise ValueError(
            f"Checkpoint missing 'config' key.\n"
            f"Cannot initialize model without config."
        )
    
    config = checkpoint['config']
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"✅ Loaded config from checkpoint (epoch {epoch})")
    
    return config


# Whitelist of safe parameters to override
SAFE_OVERRIDE_PARAMS = {
    'training.epochs': 'Extend training duration',
    'training.optimizer.lr': 'Adjust learning rate',
    'training.batch_size': 'Change batch size (memory constraint)',
    'training.gradient_accumulation_steps': 'Adjust gradient accumulation',
    'experiment.device': 'Switch device (cuda/cpu)',
    'experiment.seed': 'Change random seed',
    'training.validate_every': 'Change validation frequency',
    'training.save_every': 'Change checkpoint frequency',
    'training.evaluate_every': 'Change evaluation frequency',
    'training.save_samples_every': 'Change sample generation frequency',
    'training.num_samples': 'Change number of samples',
}


def apply_config_overrides(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
    allow_unsafe: bool = False
) -> Dict[str, Any]:
    """
    Apply config overrides with whitelist validation.
    
    Args:
        base_config: Config from checkpoint
        overrides: Dict of param paths to new values
                  e.g., {'training.epochs': 200, 'training.optimizer.lr': 5e-5}
        allow_unsafe: If True, allow overriding any parameter (dangerous!)
        
    Returns:
        Merged config
        
    Raises:
        ValueError: If trying to override unsafe parameter
    """
    config = copy.deepcopy(base_config)
    
    for param_path, new_value in overrides.items():
        # Check if parameter is in whitelist
        if not allow_unsafe and param_path not in SAFE_OVERRIDE_PARAMS:
            raise ValueError(
                f"Cannot override parameter: {param_path}\n"
                f"This parameter is not in the safe override whitelist.\n\n"
                f"Safe parameters:\n" +
                "\n".join(f"  - {k}: {v}" for k, v in SAFE_OVERRIDE_PARAMS.items()) +
                f"\n\nTo override anyway (dangerous!), use --allow-unsafe-overrides"
            )
        
        # Apply override
        try:
            old_value = get_nested_config(config, param_path)
            set_nested_config(config, param_path, new_value)
            logger.info(f"Override: {param_path} = {old_value} → {new_value}")
        except KeyError as e:
            raise ValueError(f"Config path not found: {param_path}")
    
    return config


def get_nested_config(config: Dict, path: str) -> Any:
    """Get nested config value by dot-separated path."""
    keys = path.split('.')
    value = config
    for key in keys:
        if key not in value:
            raise KeyError(f"Key '{key}' not found in config path '{path}'")
        value = value[key]
    return value


def set_nested_config(config: Dict, path: str, value: Any) -> None:
    """Set nested config value by dot-separated path."""
    keys = path.split('.')
    target = config
    for key in keys[:-1]:
        if key not in target:
            raise KeyError(f"Key '{key}' not found in config path '{path}'")
        target = target[key]
    target[keys[-1]] = value
