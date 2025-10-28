"""Complete T2F Model that orchestrates all components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Dict, Any

from .text_encoder import TextEncoder
from .mapper import TextToLatentMapper
from .stylegan_wrapper import StyleGAN2Wrapper


class T2FModel(nn.Module):
    """
    Complete Text-to-Face generation model.

    Orchestrates text encoder, mapper, StyleGAN2 generator, and loss computation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        model_cfg = config.get("model", {})
        text_encoder_cfg = model_cfg.get("text_encoder", {})
        mapper_cfg = model_cfg.get("mapper", {})
        stylegan_cfg = model_cfg.get("stylegan2", {})

        # Get latent space type
        self.latent_space = model_cfg.get("latent_space", "wplus").lower()

        # Text encoder (on CPU to save GPU memory)
        encoder_device = torch.device(text_encoder_cfg.get("device", "cpu"))
        self.text_encoder = TextEncoder(
            model_name=text_encoder_cfg.get("model_name", "bert-base-uncased"),
            embedding_dim=text_encoder_cfg.get("embedding_dim", 768),
            max_length=text_encoder_cfg.get("max_length", 128),
            freeze=text_encoder_cfg.get("freeze", True),
            pooling=text_encoder_cfg.get("pooling", "cls"),
        ).to(encoder_device)
        self.text_encoder_device = encoder_device

        # Text to latent mapper (trainable)
        self.mapper = TextToLatentMapper(
            input_dim=mapper_cfg.get("input_dim", text_encoder_cfg.get("embedding_dim", 768)),
            intermediate_dims=mapper_cfg.get("intermediate_dims", [1024, 1024]),
            output_dim=mapper_cfg.get("output_dim", 512),
            w_plus_layers=stylegan_cfg.get("w_plus_layers", 18),
            latent_space=self.latent_space,
            activation=mapper_cfg.get("activation", "relu"),
            dropout=mapper_cfg.get("dropout", 0.1),
            use_batch_norm=mapper_cfg.get("use_batch_norm", False),
            use_layer_norm=mapper_cfg.get("use_layer_norm", True),
        ).to(self.device)

        # StyleGAN2 wrapper (frozen)
        self.generator = StyleGAN2Wrapper(
            model_path=stylegan_cfg["model_path"],
            device=str(self.device),
            truncation_psi=stylegan_cfg.get("truncation_psi", 0.7),
            noise_mode=stylegan_cfg.get("noise_mode", "const"),
            stylegan_root=stylegan_cfg.get("stylegan_root"),
            compute_w_center=stylegan_cfg.get("compute_w_center", False),
            w_center_samples=stylegan_cfg.get("w_center_samples", 1000),
            w_center_batch_size=stylegan_cfg.get("w_center_batch_size", 50),
            latent_space=self.latent_space,
        )
        self.generator_device = self.generator.device
        self.truncation_psi = stylegan_cfg.get("truncation_psi", 0.7)

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text to embeddings."""
        embeddings = self.text_encoder(texts)
        return embeddings.to(self.device)

    def map_to_latent(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Map text embeddings to latent codes."""
        return self.mapper(text_embeddings.to(self.device))

    def generate_images(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """Generate images from latent codes."""
        latent_codes = latent_codes.to(self.generator_device)
        return self.generator.generate_from_latent(latent_codes)

    def forward(
        self,
        texts: Union[str, List[str]],
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Complete forward pass: text -> latent -> image.

        Args:
            texts: Input text(s)
            return_latents: Whether to return latent codes

        Returns:
            Generated images, optionally with latent codes
        """
        text_embeddings = self.encode_text(texts)
        latent_codes = self.map_to_latent(text_embeddings)

        # Ensure float32
        if latent_codes.dtype != torch.float32:
            latent_codes = latent_codes.float()

        # Apply truncation for W/W+ spaces
        if self.latent_space != "z" and self.truncation_psi < 1.0:
            latent_codes = self.generator.truncate_w(latent_codes, self.truncation_psi)

        images = self.generate_images(latent_codes)

        if return_latents:
            return images, latent_codes
        return images

    def get_trainable_parameters(self):
        """Get trainable parameters (only mapper)."""
        return [p for p in self.mapper.parameters() if p.requires_grad]

    def train_mode(self):
        """Set model to training mode (only mapper trains)."""
        self.mapper.train()
        self.text_encoder.eval()
        self.generator.eval()

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.mapper.eval()
        self.text_encoder.eval()
        self.generator.eval()
