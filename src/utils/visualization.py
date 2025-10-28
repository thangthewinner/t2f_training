"""Visualization utilities for training curves and generated images."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
    figsize: tuple = (12, 5)
):
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'epochs' keys
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    epochs = history.get('epochs', list(range(len(history['train_loss']))))
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    val_epochs = history.get('val_epochs', [])

    # Plot training loss
    if train_loss:
        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)

    # Plot validation loss (use tracked val_epochs)
    if val_loss and val_epochs:
        ax.plot(val_epochs, val_loss, 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.close(fig)


def plot_loss_components(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Loss Components",
    figsize: tuple = (12, 8)
):
    """
    Plot individual loss components (content, style, TV) over epochs.

    Args:
        history: Dictionary with 'content_loss', 'style_loss', 'tv_loss', 'epochs' keys
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    epochs = history.get('epochs', list(range(len(history.get('content_loss', [])))))
    content_loss = history.get('content_loss', [])
    style_loss = history.get('style_loss', [])
    tv_loss = history.get('tv_loss', [])

    # Plot content loss
    if content_loss:
        axes[0].plot(epochs, content_loss, 'b-', label='Content Loss', linewidth=2)
        axes[0].set_ylabel('Content Loss', fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

    # Plot style loss
    if style_loss:
        axes[1].plot(epochs, style_loss, 'r-', label='Style Loss', linewidth=2)
        axes[1].set_ylabel('Style Loss', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    # Plot TV loss
    if tv_loss:
        axes[2].plot(epochs, tv_loss, 'g-', label='TV Loss', linewidth=2)
        axes[2].set_ylabel('TV Loss', fontsize=11)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss components plot to {save_path}")

    plt.close(fig)


def plot_images(
    images: torch.Tensor,
    captions: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    nrow: int = 4,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None
):
    """
    Plot a grid of generated images with optional captions.

    Args:
        images: Tensor of shape (B, C, H, W) in range [-1, 1]
        captions: List of caption strings (optional)
        save_path: Path to save figure (optional)
        nrow: Number of images per row
        figsize: Figure size (auto-computed if None)
        title: Overall title for the plot
    """
    # Convert to numpy and denormalize
    images = images.detach().cpu()
    images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    images = images.clamp(0, 1)
    images = images.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)

    n_images = len(images)
    ncol = min(nrow, n_images)
    nrow_actual = (n_images + ncol - 1) // ncol

    if figsize is None:
        figsize = (ncol * 3, nrow_actual * 3)

    fig, axes = plt.subplots(nrow_actual, ncol, figsize=figsize)

    if nrow_actual == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow_actual == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_images):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]

        ax.imshow(images[idx])
        ax.axis('off')

        if captions and idx < len(captions):
            caption = captions[idx]
            if len(caption) > 50:
                caption = caption[:47] + "..."
            ax.set_title(caption, fontsize=8, wrap=True)

    # Hide empty subplots
    for idx in range(n_images, nrow_actual * ncol):
        row = idx // ncol
        col = idx % ncol
        axes[row, col].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved images to {save_path}")

    plt.close(fig)


def save_image_grid(
    images: torch.Tensor,
    save_path: Union[str, Path],
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True
):
    """
    Save a grid of images as a single image file.

    Args:
        images: Tensor of shape (B, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize from [-1, 1] to [0, 1]
    """
    from torchvision.utils import make_grid

    if normalize:
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)

    grid = make_grid(images, nrow=nrow, padding=padding, normalize=False)
    grid = grid.cpu().permute(1, 2, 0).numpy()
    grid = (grid * 255).astype(np.uint8)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(grid)
    img.save(save_path)
    print(f"Saved image grid to {save_path}")


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """
    Convert a tensor image to PIL Image.

    Args:
        image: Tensor of shape (C, H, W) in range [-1, 1]

    Returns:
        PIL Image
    """
    image = image.detach().cpu()
    image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)
