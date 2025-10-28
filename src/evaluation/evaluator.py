"""Model evaluator for comprehensive evaluation."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm.auto import tqdm
import json
import numpy as np

from ..models.t2f_model import T2FModel
from ..utils.memory import clear_memory
from ..utils.visualization import save_image_grid
from .metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_mse,
    calculate_lpips,
    calculate_fid_from_features,
    calculate_inception_score,
    calculate_face_semantic_distance,
    calculate_face_semantic_similarity,
    InceptionV3FeatureExtractor,
    FaceNetFeatureExtractor,
)


class Evaluator:
    """
    Comprehensive model evaluator.

    Args:
        model: T2F model instance
        device: Device to run evaluation on
        enable_fid: Enable FID calculation (requires InceptionV3)
        enable_is: Enable Inception Score calculation
        enable_lpips: Enable LPIPS calculation
        enable_face_semantic: Enable Face Semantic metrics (FSD, FSS)
    """

    def __init__(
        self,
        model: T2FModel,
        device: Optional[torch.device] = None,
        enable_fid: bool = True,
        enable_is: bool = True,
        enable_lpips: bool = True,
        enable_face_semantic: bool = True,
        num_eval_samples: int = 16,
    ):
        self.model = model
        self.device = device or model.device
        self.model.eval_mode()
        self.num_eval_samples = num_eval_samples

        # Initialize feature extractors
        self.inception_extractor = None
        self.face_extractor = None
        self.lpips_fn = None

        if enable_fid or enable_is:
            print("Loading InceptionV3 for FID/IS calculation...")
            self.inception_extractor = InceptionV3FeatureExtractor(device=str(self.device))

        if enable_face_semantic:
            print("Loading FaceNet (InceptionResnetV1) for Face Semantic metrics...")
            # FaceNet outputs 512-dim embeddings (no layer config needed)
            self.face_extractor = FaceNetFeatureExtractor(
                pretrained="vggface2",
                device=str(self.device)
            )

        if enable_lpips:
            try:
                import lpips
                print("Loading LPIPS model...")
                self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
            except ImportError:
                print("WARNING: lpips package not installed. LPIPS will be skipped.")
                print("Install with: pip install lpips")
                self.lpips_fn = None

        self.enable_fid = enable_fid and self.inception_extractor is not None
        self.enable_is = enable_is and self.inception_extractor is not None
        self.enable_lpips = enable_lpips and self.lpips_fn is not None
        self.enable_face_semantic = enable_face_semantic and self.face_extractor is not None

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        save_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        save_images: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation data
            save_dir: Directory to save results
            max_samples: Maximum number of samples to evaluate (None for all)
            save_images: Whether to save generated images

        Returns:
            Dictionary of metrics
        """
        self.model.eval_mode()

        # Only store what we need based on enabled metrics
        aggregated = {}

        # Storage for advanced metrics
        all_generated = []
        all_targets = []
        all_captions = []

        # Feature storage for FID and Face Semantic
        real_inception_features = []
        generated_inception_features = []
        real_face_features = []
        generated_face_features = []

        # Storage for IS (need full generated images)
        all_generated_for_is = []

        # Storage for LPIPS (calculate in loop to save memory)
        lpips_scores = []

        num_processed = 0

        print("\nExtracting features for evaluation metrics...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            captions = batch["captions"]
            images = batch["images"].to(self.device)

            # Generate
            generated = self.model(captions)

            # Resize if needed for fair comparison
            if generated.shape != images.shape:
                generated_resized = F.interpolate(
                    generated,
                    size=images.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            else:
                generated_resized = generated

            # Extract InceptionV3 features for FID only (not IS)
            if self.enable_fid:
                real_feat = self.inception_extractor(images)
                gen_feat = self.inception_extractor(generated_resized)
                real_inception_features.append(real_feat.cpu().numpy())
                generated_inception_features.append(gen_feat.cpu().numpy())
                del real_feat, gen_feat  # Free memory immediately
                clear_memory()

            # Extract face features for Face Semantic metrics
            if self.enable_face_semantic:
                real_face_feat = self.face_extractor(images)
                gen_face_feat = self.face_extractor(generated_resized)
                real_face_features.append(real_face_feat.cpu())
                generated_face_features.append(gen_face_feat.cpu())
                del real_face_feat, gen_face_feat  # Free memory immediately
                clear_memory()

            # Store generated images for Inception Score
            if self.enable_is:
                all_generated_for_is.append(generated_resized.cpu())

            # Calculate LPIPS for this batch
            if self.enable_lpips:
                lpips_batch = self.lpips_fn(generated_resized, images).mean().item()
                lpips_scores.append(lpips_batch)

            # Store for visualization only (limit to num_eval_samples to save memory)
            if save_images and len(all_captions) < self.num_eval_samples:
                all_generated.append(generated.cpu())
                all_targets.append(images.cpu())
                all_captions.extend(captions)

            num_processed += len(captions)
            if max_samples and num_processed >= max_samples:
                break

            clear_memory()

        # Calculate FID
        if self.enable_fid and real_inception_features:
            print("\nCalculating FID...")
            real_features = np.concatenate(real_inception_features, axis=0)
            generated_features = np.concatenate(generated_inception_features, axis=0)
            fid_score = calculate_fid_from_features(real_features, generated_features)
            aggregated['fid'] = fid_score
            print(f"FID: {fid_score:.4f}")

            # Free memory
            del real_features, generated_features, real_inception_features, generated_inception_features
            clear_memory()

        # Calculate Face Semantic metrics
        if self.enable_face_semantic and real_face_features:
            print("\nCalculating Face Semantic metrics...")
            real_face_tensor = torch.cat(real_face_features, dim=0)
            gen_face_tensor = torch.cat(generated_face_features, dim=0)

            fsd = calculate_face_semantic_distance(gen_face_tensor, real_face_tensor)
            fss = calculate_face_semantic_similarity(gen_face_tensor, real_face_tensor)

            aggregated['face_semantic_distance'] = fsd
            aggregated['face_semantic_similarity'] = fss
            print(f"Face Semantic Distance (FSD): {fsd:.4f}")
            print(f"Face Semantic Similarity (FSS): {fss:.4f}")

            # Free memory
            del real_face_tensor, gen_face_tensor, real_face_features, generated_face_features
            clear_memory()

        # Calculate Inception Score
        if self.enable_is and all_generated_for_is:
            print("\nCalculating Inception Score...")
            all_generated_tensor = torch.cat(all_generated_for_is, dim=0)

            # Move to device for IS calculation
            is_mean, is_std = calculate_inception_score(
                all_generated_tensor.to(self.device),
                self.inception_extractor,
                batch_size=32,
                splits=10
            )

            aggregated['inception_score_mean'] = is_mean
            aggregated['inception_score_std'] = is_std
            print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

            # Free memory
            del all_generated_tensor, all_generated_for_is
            clear_memory()

        # Calculate LPIPS average
        if self.enable_lpips and lpips_scores:
            print("\nCalculating LPIPS...")
            avg_lpips = float(np.mean(lpips_scores))
            aggregated['lpips'] = avg_lpips
            print(f"LPIPS: {avg_lpips:.4f}")

        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics JSON
            metrics_file = save_dir / "metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(aggregated, f, indent=2)
            print(f"\nMetrics saved to {metrics_file}")

            # Save images (generate one at a time to avoid OOM)
            if save_images and all_captions:
                print(f"\nGenerating {len(all_captions)} comparison images (one at a time to avoid OOM)...")

                # Get real images and captions (already collected during feature extraction)
                all_targets_tensor = torch.cat(all_targets, dim=0) if all_targets else None
                num_samples = min(self.num_eval_samples, len(all_captions))

                # Generate images ONE AT A TIME
                generated_images_list = []
                for i, caption in enumerate(all_captions[:num_samples]):
                    try:
                        gen_img = self.model([caption])  # Generate 1 image at a time
                        generated_images_list.append(gen_img.cpu())
                        clear_memory()
                    except Exception as e:
                        print(f"Warning: Failed to generate sample {i+1}: {e}")
                        # Use a black image as fallback
                        generated_images_list.append(torch.zeros(1, 3, 1024, 1024))

                all_generated_tensor = torch.cat(generated_images_list, dim=0)
                clear_memory()

                # Resize all images to the same size (use generated size as target)
                target_size = all_generated_tensor.shape[-1]  # Usually 1024
                if all_targets_tensor is not None and all_targets_tensor.shape[-1] != target_size:
                    all_targets_tensor = F.interpolate(
                        all_targets_tensor[:num_samples],
                        size=(target_size, target_size),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    all_targets_tensor = all_targets_tensor[:num_samples] if all_targets_tensor is not None else None

                # Create comparison grid: [target_1, target_2, gen_1, gen_2, target_3, target_4, gen_3, gen_4, ...]
                # Layout: 2 targets + 2 generated per row
                comparison_images = []
                for i in range(0, num_samples, 2):
                    # Add target images
                    if all_targets_tensor is not None:
                        comparison_images.append(all_targets_tensor[i])
                        if i + 1 < num_samples:
                            comparison_images.append(all_targets_tensor[i + 1])
                        else:
                            # Pad with zeros if odd number
                            comparison_images.append(torch.zeros_like(all_targets_tensor[i]))

                    # Add generated images
                    comparison_images.append(all_generated_tensor[i])
                    if i + 1 < num_samples:
                        comparison_images.append(all_generated_tensor[i + 1])
                    else:
                        # Pad with zeros if odd number
                        comparison_images.append(torch.zeros_like(all_generated_tensor[i]))

                comparison_tensor = torch.stack(comparison_images)

                # Save comparison grid (4 images per row: 2 targets + 2 generated)
                save_image_grid(
                    comparison_tensor,
                    save_dir / "comparison.png",
                    nrow=4
                )
                print(f"Saved comparison grid to {save_dir / 'comparison.png'}")

                # Save captions to text file
                captions_file = save_dir / "captions.txt"
                with open(captions_file, 'w', encoding='utf-8') as f:
                    f.write("Evaluation Sample Captions\n")
                    f.write("=" * 60 + "\n")
                    f.write(f"Total samples: {num_samples}\n")
                    f.write("Layout: [Target Images] | [Generated Images]\n")
                    f.write("=" * 60 + "\n\n")
                    for i, caption in enumerate(all_captions[:num_samples], 1):
                        f.write(f"Sample {i}:\n")
                        f.write(f"  {caption}\n\n")
                print(f"Saved captions to {captions_file}")

                # Clean up
                del all_generated_tensor, all_targets_tensor, comparison_tensor, generated_images_list
                clear_memory()

            print(f"Evaluation results saved to {save_dir}")

        return aggregated

    @torch.no_grad()
    def generate_samples(
        self,
        captions: List[str],
        save_dir: Optional[Path] = None,
        save_individually: bool = False,
    ) -> torch.Tensor:
        """
        Generate images from captions.

        Args:
            captions: List of text captions
            save_dir: Directory to save generated images
            save_individually: Save each image separately

        Returns:
            Generated images tensor
        """
        self.model.eval_mode()

        generated = self.model(captions)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save grid
            save_image_grid(
                generated,
                save_dir / "generated_grid.png",
                nrow=min(4, len(captions))
            )

            # Save individually
            if save_individually:
                for idx, (img, cap) in enumerate(zip(generated, captions)):
                    # Create safe filename from caption
                    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in cap)
                    safe_name = safe_name[:50]  # Limit length
                    save_image_grid(
                        img.unsqueeze(0),
                        save_dir / f"{idx:03d}_{safe_name}.png",
                        nrow=1
                    )

            print(f"\nGenerated images saved to {save_dir}")

        return generated

    @torch.no_grad()
    def interpolate_latents(
        self,
        caption1: str,
        caption2: str,
        steps: int = 8,
        save_dir: Optional[Path] = None,
    ) -> torch.Tensor:
        """
        Interpolate between two text captions in latent space.

        Args:
            caption1: First caption
            caption2: Second caption
            steps: Number of interpolation steps
            save_dir: Directory to save results

        Returns:
            Generated images tensor
        """
        self.model.eval_mode()

        # Encode texts
        emb1 = self.model.encode_text([caption1])
        emb2 = self.model.encode_text([caption2])

        # Map to latent
        latent1 = self.model.map_to_latent(emb1)
        latent2 = self.model.map_to_latent(emb2)

        # Interpolate
        alphas = torch.linspace(0, 1, steps).to(self.device)
        interpolated = []

        for alpha in alphas:
            latent = (1 - alpha) * latent1 + alpha * latent2
            img = self.model.generate_images(latent.float())
            interpolated.append(img)

        interpolated = torch.cat(interpolated, dim=0)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            save_image_grid(
                interpolated,
                save_dir / "interpolation.png",
                nrow=steps
            )

            print(f"\nInterpolation saved to {save_dir}")

        return interpolated

    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print evaluation metrics."""
        print("\n" + "=" * 50)
        print("Evaluation Metrics")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key.upper():10s}: {value:.4f}")
        print("=" * 50 + "\n")
