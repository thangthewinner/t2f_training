"""Evaluation metrics for image generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import linalg
import torchvision.models as models
from torchvision import transforms


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted images in range [-1, 1], shape (B, C, H, W)
        target: Target images in range [-1, 1], shape (B, C, H, W)

    Returns:
        PSNR value in dB
    """
    # Convert to [0, 1] range
    pred = (pred + 1.0) / 2.0
    target = (target + 1.0) / 2.0

    mse = F.mse_loss(pred, target).item()

    if mse == 0:
        return float('inf')

    return 20 * np.log10(1.0) - 10 * np.log10(mse)


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> float:
    """
    Calculate Structural Similarity Index (SSIM).

    Args:
        pred: Predicted images in range [-1, 1], shape (B, C, H, W)
        target: Target images in range [-1, 1], shape (B, C, H, W)
        window_size: Size of sliding window
        size_average: Whether to average over batch

    Returns:
        SSIM value (between -1 and 1, closer to 1 is better)
    """
    # Convert to [0, 1] range
    pred = (pred + 1.0) / 2.0
    target = (target + 1.0) / 2.0

    # Create Gaussian window
    def create_window(window_size: int, channel: int):
        def gaussian(window_size, sigma):
            gauss = torch.exp(
                -torch.arange(window_size).float().sub(window_size // 2).pow(2)
                / (2 * sigma**2)
            )
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3]).mean().item()


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        pred: Predicted images, shape (B, C, H, W)
        target: Target images, shape (B, C, H, W)

    Returns:
        MAE value
    """
    return F.l1_loss(pred, target).item()


def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        pred: Predicted images, shape (B, C, H, W)
        target: Target images, shape (B, C, H, W)

    Returns:
        MSE value
    """
    return F.mse_loss(pred, target).item()


def calculate_fid_statistics(
    features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and covariance of features for FID.

    Args:
        features: Feature array of shape (N, D)

    Returns:
        Tuple of (mean, covariance)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate Frechet Inception Distance (FID).

    Args:
        mu1: Mean of real features
        sigma1: Covariance of real features
        mu2: Mean of generated features
        sigma2: Covariance of generated features
        eps: Small constant for numerical stability

    Returns:
        FID score (lower is better)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return float(fid)


def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "vgg",
    device: Optional[torch.device] = None
) -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS).

    Note: This is a simplified placeholder. For production use, install and use
    the official lpips package: `pip install lpips`

    Args:
        pred: Predicted images in range [-1, 1], shape (B, C, H, W)
        target: Target images in range [-1, 1], shape (B, C, H, W)
        net: Network type ('vgg', 'alex', 'squeeze')
        device: Device to run on

    Returns:
        LPIPS distance (lower is better)
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=net).to(device or pred.device)
        with torch.no_grad():
            dist = loss_fn(pred, target).mean().item()
        return dist
    except ImportError:
        print("WARNING: lpips package not installed. Returning 0.0")
        print("Install with: pip install lpips")
        return 0.0


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 feature extractor for FID and IS calculation.
    Uses the pool3 layer (2048-dim features before final classification).
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        # Load pretrained InceptionV3 with new weights parameter
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        inception.eval()

        # Remove the final classification layers
        # We want features from the pool layer (2048-dim)
        self.inception = inception
        self.inception.fc = nn.Identity()  # Remove final FC layer

        self.inception.to(self.device)

        # Preprocessing: Inception expects [0, 1] range, then normalized
        self.preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            x: Images in range [-1, 1], shape (B, C, H, W)

        Returns:
            Features of shape (B, 2048)
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)

        # Resize to 299x299 (InceptionV3 input size)
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize
        x = self.preprocess(x)

        # Extract features
        features = self.inception(x)

        return features


class FaceNetFeatureExtractor(nn.Module):
    """
    FaceNet-based feature extractor for Face Semantic metrics (FSS/FSD).

    Uses InceptionResnetV1 pretrained on VGGFace2 to extract 512-dim face embeddings.
    This is the CORRECT feature extractor used in the paper, not VGG16.

    Args:
        pretrained: Pretrained dataset ('vggface2' or 'casia-webface')
        device: Device to run on

    Output:
        512-dim face embedding optimized for face recognition/similarity
    """

    def __init__(
        self,
        pretrained: str = "vggface2",
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)

        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is required for FaceNet feature extraction. "
                "Install with: pip install facenet-pytorch"
            )

        # Load pretrained FaceNet model
        self.facenet = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,  # We want embeddings, not classification
            device=self.device
        ).eval()

        # Freeze all parameters
        for param in self.facenet.parameters():
            param.requires_grad = False

        self.feature_dim = 512  # FaceNet output dimension

        print(f"FaceNetFeatureExtractor initialized with pretrained='{pretrained}'")
        print(f"Output feature dimension: {self.feature_dim}")

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for FaceNet.

        FaceNet expects images in range [-1, 1] and size 160x160.
        """
        # Input is already in [-1, 1], just need to resize
        if x.shape[2] != 160 or x.shape[3] != 160:
            x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)

        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embeddings using FaceNet.

        Args:
            x: Images in range [-1, 1], shape (B, C, H, W)

        Returns:
            Face embeddings of shape (B, 512)
        """
        x = self._preprocess(x)

        # Extract embeddings
        embeddings = self.facenet(x)

        return embeddings


def calculate_inception_score(
    images: torch.Tensor,
    feature_extractor: InceptionV3FeatureExtractor,
    batch_size: int = 32,
    splits: int = 10
) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS).

    Args:
        images: Generated images in range [-1, 1], shape (N, C, H, W)
        feature_extractor: InceptionV3 feature extractor
        batch_size: Batch size for processing
        splits: Number of splits for computing mean and std

    Returns:
        Tuple of (mean_is, std_is)
    """
    N = len(images)

    # Load full inception model ONCE (outside loop)
    inception_full = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception_full.eval()
    inception_full.to(feature_extractor.device)

    # Get predictions from InceptionV3
    preds = []

    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size].to(feature_extractor.device)

        # Get logits (before softmax) from inception
        batch = (batch + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        batch = batch.clamp(0, 1)

        if batch.shape[2] != 299 or batch.shape[3] != 299:
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

        batch = feature_extractor.preprocess(batch)

        # Use the inception model for classification
        with torch.no_grad():
            logits = inception_full(batch)
            if isinstance(logits, tuple):  # Handle auxiliary outputs during training mode
                logits = logits[0]

            pred = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    # Calculate IS
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)  # Marginal distribution
        scores = []

        for i in range(part.shape[0]):
            pyx = part[i, :]  # Conditional distribution
            scores.append(np.sum(pyx * np.log(pyx / (py + 1e-10) + 1e-10)))

        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))


def calculate_fid_from_features(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate FID from precomputed features.

    Args:
        real_features: Real image features, shape (N, D)
        generated_features: Generated image features, shape (M, D)
        eps: Small constant for numerical stability

    Returns:
        FID score (lower is better)
    """
    mu1, sigma1 = calculate_fid_statistics(real_features)
    mu2, sigma2 = calculate_fid_statistics(generated_features)

    return calculate_fid(mu1, sigma1, mu2, sigma2, eps)


def calculate_face_semantic_distance(
    pred_features: torch.Tensor,
    target_features: torch.Tensor
) -> float:
    """
    Calculate Face Semantic Distance (FSD) using L2 distance (Euclidean).

    Based on paper equation (1):
    FSD = (1/N) * Σ ||FGi - FGTi||

    Where N is the number of samples (batch size), and ||.|| is L2 norm.

    Args:
        pred_features: Generated face features, shape (B, D)
        target_features: Target face features, shape (B, D)

    Returns:
        FSD value (lower is better)
    """
    # L1 distance (Manhattan distance) over feature dimension, averaged over batch
    # Shape: (B, D) -> L1 norm over D -> (B,) -> mean over B -> scalar
    # fsd = torch.norm(pred_features - target_features, p=1, dim=1).mean().item()

    # # L2 distance (Euclidean norm) over feature dimension, averaged over batch
    # # Shape: (B, D) → norm over D → (B,) → mean over B → scalar
    fsd = torch.norm(pred_features - target_features, p=2, dim=1).mean().item()
    
    return fsd


def calculate_face_semantic_similarity(
    pred_features: torch.Tensor,
    target_features: torch.Tensor
) -> float:
    """
    Calculate Face Semantic Similarity (FSS) using cosine similarity.

    Args:
        pred_features: Generated face features, shape (B, D)
        target_features: Target face features, shape (B, D)

    Returns:
        FSS value (higher is better, range [-1, 1])
    """
    # Normalize features
    pred_norm = F.normalize(pred_features, p=2, dim=1)
    target_norm = F.normalize(target_features, p=2, dim=1)

    # Cosine similarity
    similarity = (pred_norm * target_norm).sum(dim=1).mean().item()

    # Return raw cosine similarity (no scaling, as per paper)
    return similarity
