"""StyleGAN2 Wrapper module."""

import sys
import importlib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class StyleGAN2Wrapper(nn.Module):
    """
    Wrapper for pretrained StyleGAN2 generator.

    Args:
        model_path: Path to StyleGAN2 .pkl checkpoint
        device: Device to load model on
        truncation_psi: Truncation trick parameter
        noise_mode: Noise mode for generation ("const", "random", "none")
        stylegan_root: Path to stylegan2-ada-pytorch directory
        compute_w_center: Whether to compute W center for truncation
        w_center_samples: Number of samples for computing W center
        w_center_batch_size: Batch size for computing W center
        latent_space: Latent space type ("z", "w", or "wplus")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        truncation_psi: float = 0.7,
        noise_mode: str = "const",
        stylegan_root: Optional[str] = None,
        compute_w_center: bool = False,
        w_center_samples: int = 100,
        w_center_batch_size: int = 10,
        latent_space: str = "wplus",
    ):
        super().__init__()
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        self.w_center = None
        self.latent_space = latent_space.lower()

        self._prepare_stylegan_modules(stylegan_root)
        self._load_model()

        if self.generator.w_dim != 512:
            print(f"Warning: Generator w_dim is {self.generator.w_dim}, adding projection")
            self.w_projection = nn.Linear(512, self.generator.w_dim).to(self.device)
        else:
            self.w_projection = None

        if compute_w_center:
            self._compute_w_center_batched(
                num_samples=w_center_samples, batch_size=w_center_batch_size
            )
        else:
            self.w_center = torch.zeros(1, 512, device=self.device)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _prepare_stylegan_modules(self, stylegan_root: Optional[str]):
        """Add StyleGAN2 directory to Python path."""
        if stylegan_root is None:
            stylegan_root = Path.cwd() / "stylegan2-ada-pytorch"
        stylegan_root = Path(stylegan_root).expanduser().resolve()

        if stylegan_root.exists() and str(stylegan_root) not in sys.path:
            sys.path.insert(0, str(stylegan_root))

        try:
            importlib.import_module("dnnlib")
            importlib.import_module("legacy")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Cannot find stylegan2-ada-pytorch modules. "
                "Clone the repo and pass the path through stylegan_root."
            ) from exc

    def _load_model(self):
        """Load pretrained StyleGAN2 generator."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        import legacy

        with open(self.model_path, "rb") as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(self.device).eval()

        self.generator = G
        self.latent_dim = G.z_dim
        self.resolution = G.img_resolution

        print(f"StyleGAN2 loaded:")
        print(f"  - z_dim: {G.z_dim}")
        print(f"  - w_dim: {G.w_dim}")
        print(f"  - num_ws (W+ layers): {G.num_ws}")
        print(f"  - img_resolution: {G.img_resolution}")

    def _compute_w_center_batched(self, num_samples: int, batch_size: int):
        """Compute center of W space for truncation trick."""
        with torch.no_grad():
            w_sum = torch.zeros(1, self.latent_dim, device=self.device)
            remaining = num_samples

            while remaining > 0:
                current = min(batch_size, remaining)
                z = torch.randn(current, self.latent_dim, device=self.device)
                w = self.generator.mapping(z, None)[:, 0, :]
                w_sum += w.sum(dim=0, keepdim=True)
                remaining -= current

            self.w_center = w_sum / num_samples

    def generate_from_latent(self, latent_codes: torch.Tensor, noise_mode: Optional[str] = None) -> torch.Tensor:
        """
        Generate images from latent codes.

        Args:
            latent_codes: Latent codes tensor
            noise_mode: Noise mode override

        Returns:
            Generated images [batch_size, 3, H, W]
        """
        noise_mode = self.noise_mode if noise_mode is None else noise_mode

        # Convert to correct device and dtype
        latent_codes = latent_codes.to(self.device, dtype=torch.float32)

        # Handle different latent spaces
        if self.latent_space == "z":
            # For Z space, pass through mapping network first
            w = self.generator.mapping(latent_codes, None)
            images = self.generator.synthesis(w, noise_mode=noise_mode)
        else:  # "w" or "wplus"
            # Apply projection if needed
            if self.w_projection is not None:
                original_shape = latent_codes.shape
                if latent_codes.dim() == 3:  # W+ [batch, layers, 512]
                    latent_codes = latent_codes.view(-1, latent_codes.shape[-1])

                latent_codes = self.w_projection(latent_codes)

                if len(original_shape) == 3:
                    latent_codes = latent_codes.view(original_shape[0], original_shape[1], -1)

            # Ensure shape is [batch, num_ws, w_dim] for synthesis
            if latent_codes.dim() == 2:  # W vector
                latent_codes = latent_codes.unsqueeze(1).repeat(1, self.generator.num_ws, 1)
            elif latent_codes.dim() == 3:  # W+
                if latent_codes.shape[1] != self.generator.num_ws:
                    # Adjust number of layers if needed
                    if latent_codes.shape[1] < self.generator.num_ws:
                        repeat_factor = self.generator.num_ws // latent_codes.shape[1]
                        remainder = self.generator.num_ws % latent_codes.shape[1]
                        latent_codes = latent_codes.repeat(1, repeat_factor, 1)
                        if remainder > 0:
                            latent_codes = torch.cat([
                                latent_codes,
                                latent_codes[:, -remainder:, :].repeat(1, 1, 1)
                            ], dim=1)
                    else:
                        latent_codes = latent_codes[:, :self.generator.num_ws, :]

            images = self.generator.synthesis(latent_codes, noise_mode=noise_mode)

        return images

    def truncate_w(self, w: torch.Tensor, truncation_psi: float) -> torch.Tensor:
        """Apply truncation trick to W latents."""
        if truncation_psi == 1.0 or self.w_center is None or self.latent_space == "z":
            return w

        if w.dim() == 2:
            return self.w_center + truncation_psi * (w - self.w_center)
        w_center_expanded = self.w_center.unsqueeze(1).expand_as(w)
        return w_center_expanded + truncation_psi * (w - w_center_expanded)
