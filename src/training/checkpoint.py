"""Checkpoint management module."""

import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class CheckpointMeta:
    """Metadata for a single checkpoint."""
    path: str
    epoch: int
    step: int
    loss: float
    metrics: Dict[str, float]


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup and best model tracking.

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_last_n: Number of recent checkpoints to keep
        save_best: Whether to track and save best checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 5,
        save_best: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best

        # Important paths
        self.best_ckpt_path = self.checkpoint_dir / "best_checkpoint.pt"
        self.latest_ckpt_path = self.checkpoint_dir / "latest_checkpoint.pt"
        self.metadata_path = self.checkpoint_dir / "metadata.json"

        # Tracking
        self.best_metric = float("inf")
        self.saved_checkpoints: List[CheckpointMeta] = []

        # Load existing metadata
        if self.metadata_path.exists():
            self._load_metadata()

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            state: State dict containing model, optimizer, etc.
            epoch: Current epoch number
            metrics: Metrics dict
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Save with epoch number
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(state, ckpt_path)

        # Update latest
        torch.save(state, self.latest_ckpt_path)

        # Update best if needed
        if is_best and self.save_best:
            torch.save(state, self.best_ckpt_path)
            self.best_metric = state.get('loss', float('inf'))

        # Add to tracking
        self.saved_checkpoints.append(
            CheckpointMeta(
                path=str(ckpt_path),
                epoch=epoch,
                step=state.get('step', 0),
                loss=state.get('loss', 0.0),
                metrics=metrics
            )
        )

        # Clean old checkpoints
        self._cleanup_old_checkpoints()

        # Update metadata file
        self._update_metadata()

        return ckpt_path

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        load_latest: bool = False,
        map_location: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Specific checkpoint path
            load_best: Load best checkpoint
            load_latest: Load latest checkpoint
            map_location: Device to map tensors to

        Returns:
            State dict
        """
        if load_best:
            if not self.best_ckpt_path.exists():
                raise FileNotFoundError("No best checkpoint found")
            checkpoint_path = str(self.best_ckpt_path)
        elif load_latest:
            if not self.latest_ckpt_path.exists():
                raise FileNotFoundError("No latest checkpoint found")
            checkpoint_path = str(self.latest_ckpt_path)
        elif checkpoint_path is None:
            if not self.saved_checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = self.saved_checkpoints[-1].path

        state = torch.load(checkpoint_path, map_location=map_location)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        return state

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        self.saved_checkpoints.sort(key=lambda cp: (cp.epoch, cp.step))

        while len(self.saved_checkpoints) > self.keep_last_n:
            old = self.saved_checkpoints.pop(0)
            old_path = Path(old.path)
            if old_path.exists() and old_path != self.best_ckpt_path:
                old_path.unlink()

    def _update_metadata(self):
        """Save metadata to JSON file."""
        data = {
            "checkpoints": [asdict(cp) for cp in self.saved_checkpoints],
            "best_checkpoint": str(self.best_ckpt_path) if self.best_ckpt_path.exists() else None,
            "best_metric": self.best_metric,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_metadata(self):
        """Load metadata from JSON file."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.saved_checkpoints = [CheckpointMeta(**item) for item in data.get("checkpoints", [])]
        self.best_metric = data.get("best_metric", float("inf"))
