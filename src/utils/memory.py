"""Memory management utilities for GPU memory optimization."""

import gc
import torch


def clear_memory():
    """Clear GPU and CPU cache to free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_memory_stats(device: torch.device = None):
    """
    Print current GPU memory statistics.

    Args:
        device: CUDA device (defaults to current device)
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

    print(f"GPU Memory Stats:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")


def reset_memory_stats(device: torch.device = None):
    """
    Reset peak memory statistics.

    Args:
        device: CUDA device (defaults to current device)
    """
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)
