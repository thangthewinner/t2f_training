"""Text-to-Latent Mapper module."""

import torch
import torch.nn as nn
from typing import List


def _activation(name: str):
    """Get activation function by name."""
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class TextToLatentMapper(nn.Module):
    """
    MLP network that maps text embeddings to StyleGAN2 latent codes.

    Args:
        input_dim: Input text embedding dimension
        intermediate_dims: List of hidden layer dimensions
        output_dim: Output latent dimension (typically 512)
        w_plus_layers: Number of layers for W+ space (typically 18 for StyleGAN2)
        latent_space: Target latent space ("z", "w", or "wplus")
        activation: Activation function name
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        input_dim: int = 768,
        intermediate_dims: List[int] = [1024, 1024],
        output_dim: int = 512,
        w_plus_layers: int = 18,
        latent_space: str = "wplus",
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.w_plus_layers = w_plus_layers
        self.latent_space = latent_space.lower()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        layers = []

        def add_block(in_dim: int, out_dim: int):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            elif use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Build network with intermediate layers
        prev_dim = input_dim
        for dim in intermediate_dims:
            add_block(prev_dim, dim)
            prev_dim = dim

        # Determine final output dimension based on latent space
        if self.latent_space == "z":
            final_output_dim = 512  # Z space dimension for StyleGAN2
        elif self.latent_space == "wplus":
            final_output_dim = self.w_plus_layers * self.output_dim
        else:  # "w"
            final_output_dim = self.output_dim

        layers.append(nn.Linear(prev_dim, final_output_dim))
        self.mapping_network = nn.Sequential(*layers)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map text embeddings to latent codes.

        Args:
            text_embeddings: Text embeddings [batch_size, input_dim]

        Returns:
            Latent codes in shape:
                - Z space: [batch_size, 512]
                - W space: [batch_size, 512]
                - W+ space: [batch_size, 18, 512]
        """
        # Save original batch size BEFORE any duplication
        original_batch_size = text_embeddings.size(0)

        # Handle batch_size=1 case for BatchNorm
        if self.use_batch_norm and not self.use_layer_norm and original_batch_size == 1:
            text_embeddings = text_embeddings.repeat(2, 1)
            w_flat = self.mapping_network(text_embeddings)
            w_flat = w_flat[:1]  # Take only first sample
        else:
            w_flat = self.mapping_network(text_embeddings)

        # Use original batch size for reshaping
        batch_size = original_batch_size

        # Reshape output based on latent space
        if self.latent_space == "z":
            return w_flat.view(batch_size, -1)
        elif self.latent_space == "wplus":
            return w_flat.view(batch_size, self.w_plus_layers, self.output_dim)
        else:  # "w"
            return w_flat.view(batch_size, self.output_dim)
