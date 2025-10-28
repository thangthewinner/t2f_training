#!/usr/bin/env python3
"""Main training script for T2F model."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime

from src.models.t2f_model import T2FModel
from src.data.dataloader import create_dataloaders
from src.training.trainer import Trainer
from src.utils.memory import clear_memory
from src.utils.logger import setup_logger
from src.utils.config_validator import validate_config, ConfigValidationError

# Setup module logger
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train T2F model")

    parser.add_argument(
        "--config",
        type=str,
        default=None,  # Changed: Optional now (not needed if using --resume-from)
        help="Path to config file (not needed if using --resume-from)"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Experiment directory (default: experiments/<exp_name>_<timestamp>)"
    )
    
    # NEW: Standalone checkpoint resume
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Resume training from a standalone checkpoint file.\n"
            "Config is automatically extracted from the checkpoint.\n"
            "Creates a new experiment directory.\n\n"
            "Example:\n"
            "  python train.py --resume-from checkpoint_epoch0050.pt\n"
        )
    )
    
    # OLD: Resume from experiment directory (keep for backward compatibility)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (requires --config, old behavior)"
    )
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume from best checkpoint (requires --config)"
    )
    
    # NEW: Config overrides
    parser.add_argument(
        "--override",
        type=str,
        nargs='*',
        default=[],
        help=(
            "Override config parameters (whitelist only).\n"
            "Format: param.path=value\n\n"
            "Examples:\n"
            "  --override training.epochs=200\n"
            "  --override training.optimizer.lr=5e-5\n"
        )
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_override_args(override_list: list) -> dict:
    """
    Parse --override arguments into dict.
    
    Example:
        ['training.epochs=200', 'training.optimizer.lr=5e-5']
        → {'training.epochs': 200, 'training.optimizer.lr': 5e-5}
    """
    overrides = {}
    for override in override_list:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}\nExpected: param.path=value")
        
        path, value = override.split('=', 1)
        
        # Try to parse value type
        try:
            # Try int
            value = int(value)
        except ValueError:
            try:
                # Try float
                value = float(value)
            except ValueError:
                # Keep as string (handle bools)
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
        
        overrides[path] = value
    
    return overrides


def main():
    """Main training function."""
    args = parse_args()

    # ==========================================
    # DETECT TRAINING MODE
    # ==========================================
    
    checkpoint_to_load = None
    
    if args.resume_from:
        # ========================================
        # MODE 2: STANDALONE CHECKPOINT (NEW)
        # ========================================
        print("\n" + "="*60)
        print("MODE: RESUME FROM STANDALONE CHECKPOINT")
        print("="*60)
        
        # Check for flag conflicts
        if args.config:
            logger.warning(
                "Both --config and --resume-from provided.\n"
                "Priority: --resume-from (config extracted from checkpoint)\n"
                "Ignoring: --config argument"
            )
        if args.resume or args.resume_best:
            logger.warning(
                "Both --resume and --resume-from provided.\n"
                "Using: --resume-from (standalone mode)"
            )
        
        # Import utilities
        from src.utils.checkpoint_utils import (
            extract_config_and_create_experiment_dir,
            apply_config_overrides
        )
        
        # Extract config and create experiment dir
        checkpoint_path = Path(args.resume_from)
        config, experiment_dir = extract_config_and_create_experiment_dir(
            checkpoint_path,
            validate_config=True
        )
        
        # Apply overrides if specified
        if args.override:
            overrides = parse_override_args(args.override)
            config = apply_config_overrides(config, overrides)
        
        # Override CLI args (device, seed)
        if args.device:
            config["experiment"]["device"] = args.device
        if args.seed is not None:
            config["experiment"]["seed"] = args.seed
        
        # Checkpoint to load
        checkpoint_to_load = checkpoint_path
        
    elif args.resume or args.resume_best:
        # ========================================
        # MODE 3: RESUME FROM EXPERIMENT DIR (OLD)
        # ========================================
        print("\n" + "="*60)
        print("MODE: RESUME FROM EXPERIMENT DIRECTORY")
        print("="*60)
        
        if not args.config:
            raise ValueError(
                "Error: --resume requires --config\n\n"
                "Did you mean to use --resume-from for standalone checkpoint?\n"
                "  python train.py --resume-from checkpoint.pt\n"
            )
        
        # Load configuration
        try:
            logger.info(f"Loading config from: {args.config}")
            config = load_config(args.config)
        except FileNotFoundError:
            logger.error(f"Config file not found: {args.config}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise
        
        # Validate configuration
        try:
            logger.info("Validating configuration...")
            validate_config(config)
            logger.info("Configuration validation passed")
        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed:\n{e}")
            raise
        
        # Override config with command-line arguments
        if args.device:
            config["experiment"]["device"] = args.device
        if args.seed is not None:
            config["experiment"]["seed"] = args.seed
        
        # Setup experiment directory (provided or default)
        if args.experiment_dir:
            experiment_dir = Path(args.experiment_dir)
        else:
            exp_name = config["experiment"].get("name", "t2f_experiment")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = Path("experiments") / f"{exp_name}_{timestamp}"
        
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine checkpoint path
        checkpoint_to_load = Path(args.resume) if args.resume else None
        
    else:
        # ========================================
        # MODE 1: FRESH TRAINING
        # ========================================
        print("\n" + "="*60)
        print("MODE: FRESH TRAINING")
        print("="*60)
        
        if not args.config:
            raise ValueError(
                "Error: --config required for fresh training\n\n"
                "Examples:\n"
                "  Fresh training:\n"
                "    python train.py --config configs/wplus_space.yaml\n\n"
                "  Resume from checkpoint:\n"
                "    python train.py --resume-from checkpoint_epoch0050.pt\n"
            )
        
        # Load configuration
        try:
            logger.info(f"Loading config from: {args.config}")
            config = load_config(args.config)
        except FileNotFoundError:
            logger.error(f"Config file not found: {args.config}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise

        # Validate configuration
        try:
            logger.info("Validating configuration...")
            validate_config(config)
            logger.info("Configuration validation passed")
        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed:\n{e}")
            raise

        # Override config with command-line arguments
        if args.device:
            config["experiment"]["device"] = args.device
        if args.seed is not None:
            config["experiment"]["seed"] = args.seed

        # Setup experiment directory
        if args.experiment_dir:
            experiment_dir = Path(args.experiment_dir)
        else:
            exp_name = config["experiment"].get("name", "t2f_experiment")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = Path("experiments") / f"{exp_name}_{timestamp}"

        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_to_load = None
    
    # ==========================================
    # COMMON: Setup and Training
    # ==========================================
    
    logger.info(f"Experiment directory: {experiment_dir}")

    # Set random seed
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    # Setup file logger
    log_file = experiment_dir / "logs" / "training.log"
    setup_logger(__name__, log_file=log_file, level=logging.INFO)
    logger.info("File logging initialized")

    # Save config to experiment directory (skip if already saved by resume mode)
    config_save_path = experiment_dir / "config.yaml"
    if not config_save_path.exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Saved config to: {config_save_path}")
    else:
        logger.info(f"Config already exists at: {config_save_path}")

    # Setup device
    device_str = config["experiment"].get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Initializing model...")
    # Add device to config for model initialization
    config_with_device = config.copy()
    config_with_device["device"] = str(device)

    try:
        model = T2FModel(config=config_with_device)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        logger.info(f"Batch size: {config['training']['batch_size']}")
        logger.info(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
        logger.info(f"Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        experiment_dir=experiment_dir,
    )
    logger.info("Trainer initialized successfully")

    # Load checkpoint if resuming
    if checkpoint_to_load:
        logger.info(f"Loading checkpoint: {checkpoint_to_load}")
        try:
            trainer.load_checkpoint(
                checkpoint_path=str(checkpoint_to_load),
                load_best=args.resume_best  # Only used in old mode
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # Start training
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    try:
        trainer.fit(train_loader, val_loader, test_loader)
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
        logger.info("Saving current state...")
        # Save checkpoint on interrupt
        state = {
            'epoch': trainer.start_epoch,
            'step': trainer.global_step,
            'model_state_dict': model.mapper.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'config': config,
            'history': trainer.history,
        }
        torch.save(state, experiment_dir / "interrupted_checkpoint.pt")
        logger.info(f"Checkpoint saved to {experiment_dir / 'interrupted_checkpoint.pt'}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {experiment_dir}")

    # Clear memory
    clear_memory()


if __name__ == "__main__":
    main()
