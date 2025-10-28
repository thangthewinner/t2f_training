"""DataLoader creation utilities."""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .dataset import T2FDataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary with images and captions
    """
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    result = {"images": images, "captions": captions}

    if "image_path" in batch[0]:
        result["image_paths"] = [item["image_path"] for item in batch]

    return result


def create_datasets(config: Dict[str, Any]) -> Tuple[T2FDataset, T2FDataset, T2FDataset]:
    """
    Create train, validation, and test datasets from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    dataset_cfg = config["dataset"]

    experiment_seed = config.get("experiment", {}).get("seed")
    dataset_seed = dataset_cfg.get("seed", experiment_seed)

    common_params = {
        "data_root": dataset_cfg["data_root"],
        "caps_file": dataset_cfg.get("caps_file", "caps.txt"),
        "img_dir": dataset_cfg.get("img_dir", "img"),
        "image_size": dataset_cfg["image_size"],
        "train_split": dataset_cfg.get("train_split", 0.7),
        "val_split": dataset_cfg.get("val_split", 0.15),
        "test_split": dataset_cfg.get("test_split", 0.15),
        "subset_size": dataset_cfg.get("subset_size"),
        "return_paths": dataset_cfg.get("return_paths", False),
        "use_augmentation": dataset_cfg.get("use_augmentation", True),
        "augmentation_config": dataset_cfg.get("augmentation", {}),
        "seed": dataset_seed,
    }

    train_dataset = T2FDataset(split="train", **common_params)
    val_dataset = T2FDataset(split="val", **common_params)
    test_dataset = T2FDataset(split="test", **common_params)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    dataset_cfg = config["dataset"]
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get("batch_size", 1),
        shuffle=dataset_cfg.get("shuffle", True),
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_cfg.get("batch_size", training_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg.get("batch_size", training_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
