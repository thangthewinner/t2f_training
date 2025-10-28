#!/usr/bin/env python3
"""Evaluation script for T2F model."""

import argparse
import yaml
import torch
from pathlib import Path

from src.models.t2f_model import T2FModel
from src.data.dataloader import create_dataloaders
from src.evaluation.evaluator import Evaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate T2F model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file or experiment directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: load best from experiment)"
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        help="Load best checkpoint from experiment directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Do not save generated images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)"
    )

    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load configuration from YAML file or experiment directory."""
    path = Path(path)

    # If path is a directory, look for config.yaml inside
    if path.is_dir():
        config_path = path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.yaml found in {path}")
    else:
        config_path = path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_checkpoint_path(args) -> str:
    """Determine checkpoint path from arguments."""
    if args.checkpoint:
        return args.checkpoint

    # Try to find checkpoint in experiment directory
    config_path = Path(args.config)
    if config_path.is_dir():
        exp_dir = config_path
        ckpt_dir = exp_dir / "checkpoints"

        if args.load_best:
            ckpt_path = ckpt_dir / "best_checkpoint.pt"
            if ckpt_path.exists():
                return str(ckpt_path)
            else:
                raise FileNotFoundError(f"No best checkpoint found in {ckpt_dir}")
        else:
            # Load latest
            ckpt_path = ckpt_dir / "latest_checkpoint.pt"
            if ckpt_path.exists():
                return str(ckpt_path)
            else:
                raise FileNotFoundError(f"No latest checkpoint found in {ckpt_dir}")

    raise ValueError("Must specify --checkpoint or provide experiment directory")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override device if specified
    if args.device:
        config["experiment"]["device"] = args.device

    # Setup device
    device_str = config["experiment"].get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\nInitializing model...")
    # Add device to config for model initialization
    config_with_device = config.copy()
    config_with_device["device"] = str(device)

    model = T2FModel(config=config_with_device)

    # Load checkpoint
    checkpoint_path = load_checkpoint_path(args)
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.mapper.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint from epoch {state.get('epoch', 'unknown')}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Select dataloader based on split
    if args.split == "train":
        dataloader = train_loader
    elif args.split == "val":
        dataloader = val_loader
    else:
        dataloader = test_loader

    print(f"Evaluating on {args.split} split ({len(dataloader.dataset)} samples)")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        config_path = Path(args.config)
        if config_path.is_dir():
            output_dir = config_path / "evaluation"
        else:
            output_dir = Path("evaluation_results")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create evaluator
    print("\nInitializing evaluator...")
    eval_cfg = config.get("evaluation", {})
    face_semantic_cfg = eval_cfg.get("face_semantic", {})
    evaluator = Evaluator(
        model=model,
        device=device,
        enable_fid=eval_cfg.get("enable_fid", True),
        enable_is=eval_cfg.get("enable_is", False),
        enable_lpips=eval_cfg.get("enable_lpips", False),
        enable_face_semantic=eval_cfg.get("enable_face_semantic", True),
        num_eval_samples=eval_cfg.get("num_eval_samples", 16),
        face_semantic_layers=face_semantic_cfg.get("layers"),
    )

    # Run evaluation
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    max_samples = args.max_samples if args.max_samples is not None else eval_cfg.get("max_samples", None)
    save_images = False if args.no_save_images else eval_cfg.get("save_images", True)

    metrics = evaluator.evaluate(
        dataloader=dataloader,
        save_dir=output_dir,
        max_samples=max_samples,
        save_images=save_images,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    # Print metrics
    evaluator.print_metrics(metrics)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
