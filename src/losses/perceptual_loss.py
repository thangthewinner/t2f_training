"""Perceptual loss module using VGG16."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict, Optional, Union, Tuple
from torchvision import models


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for style loss."""
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (c * h * w)


class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss with content, style, and TV regularization.

    Args:
        content_layers: VGG layer names for content loss
        style_layers: VGG layer names for style loss
        content_weight: Weight for content loss
        style_weight: Weight for style loss
        tv_weight: Weight for total variation loss
        device: Device to place VGG on
        vgg_input_size: Input size for VGG (resize images to this)
    """

    def __init__(
        self,
        content_layers: Iterable[str] = ("conv3_3",),
        style_layers: Iterable[str] = ("conv3_3", "conv4_3"),
        content_weight: float = 1.0,
        style_weight: float = 1e3,
        tv_weight: float = 1e-6,
        device: str = "cuda",
        vgg_input_size: int = 224,
    ):
        super().__init__()

        # Save layer names
        self.content_layer_names = tuple(content_layers)
        self.style_layer_names = tuple(style_layers)

        # Convert weights to float in case they're strings from YAML (e.g., "1e-6")
        self.content_weight = float(content_weight)
        self.style_weight = float(style_weight)
        self.tv_weight = float(tv_weight)

        self.device = torch.device(device)
        self.vgg_input_size = vgg_input_size

        # Load VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()

        # Map layer names to indices in VGG16
        # VGG16 architecture: 5 blocks with conv layers
        self.layer_name_to_idx = {
            # Block 1
            "conv1_1": 0,
            "conv1_2": 2,
            # Block 2
            "conv2_1": 5,
            "conv2_2": 7,
            # Block 3
            "conv3_1": 10,
            "conv3_2": 12,
            "conv3_3": 14,
            # Block 4
            "conv4_1": 17,
            "conv4_2": 19,
            "conv4_3": 21,
            # Block 5
            "conv5_1": 24,
            "conv5_2": 26,
            "conv5_3": 28,
        }

        # Validate layer names
        invalid_content = [name for name in self.content_layer_names if name not in self.layer_name_to_idx]
        invalid_style = [name for name in self.style_layer_names if name not in self.layer_name_to_idx]

        if invalid_content:
            raise ValueError(
                f"Invalid content layer names: {invalid_content}. "
                f"Valid layers: {list(self.layer_name_to_idx.keys())}"
            )
        if invalid_style:
            raise ValueError(
                f"Invalid style layer names: {invalid_style}. "
                f"Valid layers: {list(self.layer_name_to_idx.keys())}"
            )

        # Get indices of layers we care about
        self.content_layer_idx = {self.layer_name_to_idx[name] for name in self.content_layer_names}
        self.style_layer_idx = {self.layer_name_to_idx[name] for name in self.style_layer_names}

        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(self.device)

        # ImageNet normalization
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess image for VGG (normalize to ImageNet stats)."""
        x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        return (x - self.imagenet_mean) / self.imagenet_std

    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified VGG layers."""
        feats = {}
        x_norm = self._preprocess(x.to(self.device))

        # Resize to VGG input size
        if x_norm.shape[2] != self.vgg_input_size or x_norm.shape[3] != self.vgg_input_size:
            x_norm = F.interpolate(x_norm, size=(self.vgg_input_size, self.vgg_input_size), mode="bilinear", align_corners=False)

        # Forward through VGG and save features
        for i, layer in enumerate(self.vgg):
            x_norm = layer(x_norm)

            if i in self.content_layer_idx or i in self.style_layer_idx:
                layer_name = next(name for name, idx in self.layer_name_to_idx.items() if idx == i)
                feats[layer_name] = x_norm.clone()

        return feats

    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss for smoothness."""
        loss_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        loss_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return loss_h + loss_w

    def tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public method to compute total variation loss.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Total variation loss (scalar tensor)
        """
        return self._total_variation(x)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        style_target: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted images
            target: Target images
            style_target: Target for style loss (defaults to target)
            return_components: Whether to return loss components

        Returns:
            Total loss, optionally with components dict
        """
        if style_target is None:
            style_target = target

        pred_feats = self._extract_features(pred)
        target_feats = self._extract_features(target)
        style_feats = self._extract_features(style_target)

        # Content loss
        content_loss = 0.0
        for layer_name in self.content_layer_names:
            content_loss += F.mse_loss(pred_feats[layer_name], target_feats[layer_name])
        content_loss *= self.content_weight

        # Style loss (Gram matrix)
        style_loss = 0.0
        if self.style_weight > 0 and len(self.style_layer_names) > 0:
            for layer_name in self.style_layer_names:
                gram_pred = gram_matrix(pred_feats[layer_name])
                gram_style = gram_matrix(style_feats[layer_name])
                style_loss += F.mse_loss(gram_pred, gram_style)
            style_loss *= self.style_weight

        # Total variation loss
        tv_loss = self._total_variation(pred) * self.tv_weight if self.tv_weight > 0 else 0.0

        # Convert to tensors if they are floats
        if isinstance(content_loss, float):
            content_loss = torch.tensor(content_loss, device=self.device)
        if isinstance(style_loss, float):
            style_loss = torch.tensor(style_loss, device=self.device)
        if isinstance(tv_loss, float):
            tv_loss = torch.tensor(tv_loss, device=self.device)

        total_loss = content_loss + style_loss + tv_loss

        # Ensure gradient flow
        if not total_loss.requires_grad:
            dummy_term = torch.tensor(0.0, device=self.device, requires_grad=True)
            total_loss = total_loss + dummy_term

        if return_components:
            # Safely convert to items (handle both tensors and floats)
            content_val = content_loss.item() if torch.is_tensor(content_loss) else float(content_loss)
            style_val = style_loss.item() if torch.is_tensor(style_loss) else float(style_loss)
            tv_val = tv_loss.item() if torch.is_tensor(tv_loss) else float(tv_loss)

            return total_loss, {
                "content": content_val,
                "style": style_val,
                "tv": tv_val,
                "total": total_loss.item(),
            }
        return total_loss
