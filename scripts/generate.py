#!/usr/bin/env python3
"""Generation script for T2F model."""

import argparse
import yaml
import torch
from pathlib import Path
from typing import List

from src.models.t2f_model import T2FModel
from src.evaluation.evaluator import Evaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate images from text using T2F model")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file or experiment directory (optional if checkpoint provided)"
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
        "--captions",
        type=str,
        nargs="+",
        default=None,
        help="Text captions for generation"
    )
    parser.add_argument(
        "--captions-file",
        type=str,
        default=None,
        help="File containing captions (one per line)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--save-individually",
        action="store_true",
        help="Save each image separately"
    )
    parser.add_argument(
        "--interpolate",
        nargs=2,
        metavar=("CAPTION1", "CAPTION2"),
        default=None,
        help="Interpolate between two captions"
    )
    parser.add_argument(
        "--interpolate-steps",
        type=int,
        default=8,
        help="Number of interpolation steps"
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


def load_captions(args) -> List[str]:
    """Load captions from command line or file."""
    if args.captions:
        return args.captions
    elif args.captions_file:
        with open(args.captions_file, "r", encoding="utf-8") as f:
            captions = [line.strip() for line in f if line.strip()]
        return captions
    else:
        # Default captions
        return [
            "A young woman with long blonde hair and blue eyes",
            "A middle-aged man with short brown hair and a beard",
            "A person with curly red hair and freckles",
            "A smiling woman with dark hair and glasses",
        ]


def main():
    """Main generation function."""
    args = parse_args()

    # ==========================================
    # LOAD CONFIG (Auto-detect from checkpoint)
    # ==========================================
    
    if args.config:
        # Explicit config provided
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        # Auto-extract from checkpoint
        if not args.checkpoint:
            raise ValueError(
                "Error: Must provide either --config or --checkpoint\n\n"
                "Examples:\n"
                "  With config:\n"
                "    python generate.py --config cfg.yaml --checkpoint ckpt.pt\n\n"
                "  Without config (auto-extract):\n"
                "    python generate.py --checkpoint ckpt.pt --captions \"A woman\"\n"
            )
        
        print("No config provided, extracting from checkpoint...")
        from src.utils.checkpoint_utils import load_config_from_checkpoint
        
        checkpoint_path = Path(args.checkpoint)
        config = load_config_from_checkpoint(checkpoint_path)
        print("✅ Config extracted from checkpoint")

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

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create evaluator
    evaluator = Evaluator(model=model, device=device)

    # Generate images
    if args.interpolate:
        # Interpolation mode
        caption1, caption2 = args.interpolate
        print("\n" + "=" * 60)
        print("INTERPOLATING BETWEEN CAPTIONS")
        print("=" * 60)
        print(f"Caption 1: {caption1}")
        print(f"Caption 2: {caption2}")
        print(f"Steps: {args.interpolate_steps}")

        evaluator.interpolate_latents(
            caption1=caption1,
            caption2=caption2,
            steps=args.interpolate_steps,
            save_dir=output_dir,
        )

    else:
        # Normal generation mode
        captions = load_captions(args)

        print("\n" + "=" * 60)
        print("GENERATING IMAGES")
        print("=" * 60)
        print(f"Number of captions: {len(captions)}")
        print("\nCaptions:")
        for i, cap in enumerate(captions, 1):
            print(f"  {i}. {cap}")

        evaluator.generate_samples(
            captions=captions,
            save_dir=output_dir,
            save_individually=args.save_individually,
        )

    print("\n" + "=" * 60)
    print("GENERATION COMPLETED")
    print("=" * 60)
    print(f"\nImages saved to: {output_dir}")


if __name__ == "__main__":
    main()
