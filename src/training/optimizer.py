"""Optimizer and scheduler creation utilities."""

import torch
from typing import Dict, Any, Optional, Iterable


def create_optimizer(
    parameters: Iterable,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        parameters: Model parameters to optimize
        config: Optimizer configuration

    Returns:
        Optimizer instance
    """
    opt_type = config.get("type", "adam").lower()
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 0.0)

    if not isinstance(parameters, (list, tuple)):
        parameters = list(parameters)

    if opt_type == "adam":
        betas = config.get("betas", (0.9, 0.999))
        return torch.optim.Adam(
            parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    elif opt_type == "adamw":
        betas = config.get("betas", (0.9, 0.999))
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    elif opt_type == "sgd":
        momentum = config.get("momentum", 0.9)
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration

    Returns:
        Scheduler instance or None
    """
    if "scheduler" not in config:
        return None

    sched_cfg = config["scheduler"]
    sched_type = sched_cfg.get("type", "cosine").lower()

    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", 100),
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 50),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 10),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    elif sched_type == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
