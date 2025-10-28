"""Main trainer module for T2F training."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm.auto import tqdm

from ..models.t2f_model import T2FModel
from ..losses.perceptual_loss import PerceptualLoss
from ..utils.memory import clear_memory
from ..utils.visualization import plot_training_curves, plot_loss_components, save_image_grid
from .checkpoint import CheckpointManager
from .optimizer import create_optimizer, create_scheduler
import json


class Trainer:
    """
    Main trainer for T2F model.

    Args:
        model: T2F model instance
        config: Training configuration
        experiment_dir: Directory to save experiment results
    """

    def __init__(
        self,
        model: T2FModel,
        config: Dict[str, Any],
        experiment_dir: Path
    ):
        self.model = model
        self.config = config
        self.device = model.device
        self.experiment_dir = Path(experiment_dir)

        # Create subdirectories
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        self.samples_dir = self.experiment_dir / "samples"
        self.plots_dir = self.experiment_dir / "plots"

        for dir_path in [self.checkpoint_dir, self.log_dir, self.samples_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Training config
        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 100)
        self.validate_every = training_cfg.get("validate_every", 5)
        self.save_every = training_cfg.get("save_every", 10)
        self.log_interval = training_cfg.get("log_interval", 10)
        self.grad_clip = training_cfg.get("grad_clip", 1.0)
        self.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 4)

        # Evaluation during training config
        self.evaluate_every = training_cfg.get("evaluate_every", None)  # None = only at end
        self.run_final_evaluation = training_cfg.get("run_final_evaluation", True)

        # Loss function
        loss_cfg = config.get("model", {}).get("loss", {})
        self.criterion = PerceptualLoss(
            content_layers=loss_cfg.get("content_layers", ("conv3_3",)),
            style_layers=loss_cfg.get("style_layers", ("conv3_3", "conv4_3")),
            content_weight=loss_cfg.get("content_weight", 1.0),
            style_weight=loss_cfg.get("style_weight", 0.0),
            tv_weight=loss_cfg.get("tv_weight", 1e-6),
            device=str(self.device),
            vgg_input_size=loss_cfg.get("vgg_input_size", 128),
        )

        # Optimizer and scheduler
        self.optimizer = create_optimizer(
            model.get_trainable_parameters(),
            training_cfg.get("optimizer", {})
        )
        self.scheduler = create_scheduler(self.optimizer, training_cfg.get("optimizer", {}))

        # Checkpoint manager
        self.ckpt_mgr = CheckpointManager(
            self.checkpoint_dir,
            keep_last_n=training_cfg.get("keep_last_n", 5)
        )

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'val_epochs': [],  # Track which epochs have validation
            # Loss components
            'content_loss': [],
            'style_loss': [],
            'tv_loss': [],
        }

        # Sample generation settings
        self.save_samples_every = training_cfg.get("save_samples_every", self.save_every)
        self.num_samples = training_cfg.get("num_samples", 4)
        # self.sample_captions = training_cfg.get("sample_captions", [
        #     "A young woman with long blonde hair and blue eyes",
        #     "A middle-aged man with short brown hair and a beard",
        #     "A person with curly red hair and freckles",
        #     "A smiling woman with dark hair and glasses",
        # ])

        # Fixed samples for consistent tracking across epochs
        self.fixed_samples = None  # Will be set on first call to generate_samples

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> tuple:
        """
        Train for one epoch.

        Returns:
            (avg_loss, avg_content, avg_style, avg_tv, avg_grad_norm, max_grad_norm)
        """
        self.model.train_mode()
        self.optimizer.zero_grad()

        running_loss = 0.0
        running_content = 0.0
        running_style = 0.0
        running_tv = 0.0
        running_grad_norm = 0.0
        max_grad_norm = 0.0
        grad_norm_count = 0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch_idx, batch in enumerate(progress):
            captions = batch["captions"]
            images = batch["images"].to(self.device)

            # Forward pass
            text_embeddings = self.model.encode_text(captions)
            latent_codes = self.model.map_to_latent(text_embeddings)
            generated = self.model.generate_images(latent_codes.float())

            # Compute loss
            target_size = self.criterion.vgg_input_size
            if generated.shape[-1] > target_size:
                generated_resized = F.interpolate(
                    generated, size=(target_size, target_size),
                    mode="bilinear", align_corners=False
                )
            else:
                generated_resized = generated

            if images.shape[-1] != target_size:
                images_resized = F.interpolate(
                    images, size=(target_size, target_size),
                    mode="bilinear", align_corners=False
                )
            else:
                images_resized = images

            # Get loss with components
            loss, loss_components = self.criterion(
                generated_resized,
                images_resized,
                return_components=True
            )

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item() * self.gradient_accumulation_steps
            running_content += loss_components['content']
            running_style += loss_components['style']
            running_tv += loss_components['tv']
            num_batches += 1

            # Optimizer step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Compute gradient norm BEFORE clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    max_norm=float('inf')  # Just compute, don't clip yet
                ).item()

                running_grad_norm += grad_norm
                max_grad_norm = max(max_grad_norm, grad_norm)
                grad_norm_count += 1

                # Now actually clip if needed
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.grad_clip
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Update progress bar with gradient info
            if batch_idx % self.log_interval == 0:
                avg_grad = running_grad_norm / max(1, grad_norm_count)
                progress.set_postfix({
                    "loss": f"{running_loss/max(1, num_batches):.4f}",
                    "grad_norm": f"{avg_grad:.3f}"
                })

            # Clear memory
            del generated, generated_resized, images_resized, loss
            clear_memory()

        # Flush remaining gradients if last batch didn't trigger step
        if num_batches % self.gradient_accumulation_steps != 0:
            # Compute gradient norm for the final partial batch
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                max_norm=float('inf')
            ).item()

            running_grad_norm += grad_norm
            max_grad_norm = max(max_grad_norm, grad_norm)
            grad_norm_count += 1

            # Clip if needed
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    self.grad_clip
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        avg_loss = running_loss / max(1, num_batches)
        avg_content = running_content / max(1, num_batches)
        avg_style = running_style / max(1, num_batches)
        avg_tv = running_tv / max(1, num_batches)
        avg_grad_norm = running_grad_norm / max(1, grad_norm_count)

        return avg_loss, avg_content, avg_style, avg_tv, avg_grad_norm, max_grad_norm

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval_mode()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validating"):
            captions = batch["captions"]
            images = batch["images"].to(self.device)

            # Forward pass
            generated = self.model(captions)

            # Compute loss
            target_size = self.criterion.vgg_input_size
            if generated.shape[-1] > target_size:
                generated = F.interpolate(
                    generated, size=(target_size, target_size),
                    mode="bilinear", align_corners=False
                )
            if images.shape[-1] != target_size:
                images = F.interpolate(
                    images, size=(target_size, target_size),
                    mode="bilinear", align_corners=False
                )

            loss = self.criterion(generated, images)

            total_loss += loss.item()
            num_batches += 1

            clear_memory()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def generate_samples(self, epoch: int, dataloader: DataLoader):
        """
        Generate and save sample images with comparison to real images.

        Uses fixed samples across all epochs for consistent progress tracking.
        Creates a grid with format:
        [real_1] [real_2] | [gen_1] [gen_2]
        [real_3] [real_4] | [gen_3] [gen_4]

        Args:
            epoch: Current epoch number
            dataloader: DataLoader to get real samples from (only used once)
        """
        self.model.eval_mode()

        try:
            # Cache fixed samples on first call
            if self.fixed_samples is None:
                print("Caching fixed samples for consistent tracking across epochs...")

                # Collect samples until we have enough
                all_captions = []
                all_images = []

                dataloader_iter = iter(dataloader)
                while len(all_captions) < self.num_samples:
                    try:
                        batch = next(dataloader_iter)
                        all_captions.extend(batch["captions"])
                        all_images.append(batch["images"])
                    except StopIteration:
                        break

                # Concatenate and take exactly num_samples
                all_images = torch.cat(all_images, dim=0)[:self.num_samples]
                all_captions = all_captions[:self.num_samples]

                self.fixed_samples = {
                    'captions': all_captions,
                    'images': all_images.clone()  # Clone to avoid modification
                }
                print(f"Fixed {len(self.fixed_samples['captions'])} samples for all epochs")

            # Use cached fixed samples
            captions = self.fixed_samples['captions']
            real_images = self.fixed_samples['images'].to(self.device)

            # Generate images ONE AT A TIME to avoid OOM
            print(f"Generating {len(captions)} samples (one at a time to avoid OOM)...")
            generated_images_list = []
            for i, caption in enumerate(captions):
                try:
                    gen_img = self.model([caption])  # Generate 1 image at a time
                    generated_images_list.append(gen_img)
                    clear_memory()
                except Exception as e:
                    print(f"Warning: Failed to generate sample {i+1}: {e}")
                    # Use a black image as fallback
                    generated_images_list.append(torch.zeros(1, 3, 1024, 1024, device=self.device))

            generated_images = torch.cat(generated_images_list, dim=0)
            clear_memory()

            # Resize all images to the same size (use generated size as target)
            target_size = generated_images.shape[-1]  # Usually 1024

            # Resize real images to match generated images
            if real_images.shape[-1] != target_size:
                real_images = F.interpolate(
                    real_images,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )

            # Create grid layout: [real_1, real_2, gen_1, gen_2, real_3, real_4, gen_3, gen_4, ...]
            # This creates:
            # Row 1: [real_1] [real_2] [gen_1] [gen_2]
            # Row 2: [real_3] [real_4] [gen_3] [gen_4]
            num_samples = len(captions)
            comparison_images = []

            # Process in pairs (2 samples per row)
            for i in range(0, num_samples, 2):
                # Add real images for this row
                comparison_images.append(real_images[i])
                if i + 1 < num_samples:
                    comparison_images.append(real_images[i + 1])
                else:
                    # Pad with zeros if odd number of samples
                    comparison_images.append(torch.zeros_like(real_images[i]))

                # Add generated images for this row
                comparison_images.append(generated_images[i])
                if i + 1 < num_samples:
                    comparison_images.append(generated_images[i + 1])
                else:
                    # Pad with zeros if odd number of samples
                    comparison_images.append(torch.zeros_like(generated_images[i]))

            # Stack all images
            comparison_images = torch.stack(comparison_images)

            # Save grid (4 columns per row: 2 real + 2 generated)
            save_path = self.samples_dir / f"epoch_{epoch+1:04d}.png"
            save_image_grid(
                comparison_images,
                save_path=save_path,
                nrow=4,  # 4 images per row (2 real + 2 generated)
                normalize=True
            )

            # Save captions to text file (only once, on first epoch)
            caption_path = self.samples_dir / "captions.txt"
            if not caption_path.exists():
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write("Fixed Sample Captions (Same across all epochs)\n")
                    f.write("=" * 60 + "\n")
                    f.write("Layout: [Real Images] | [Generated Images]\n")
                    f.write("=" * 60 + "\n\n")
                    for i, caption in enumerate(captions, 1):
                        f.write(f"Sample {i}:\n")
                        f.write(f"  Caption: {caption}\n\n")
                print(f"Saved captions to {caption_path}")

            print(f"Saved comparison samples to {save_path}")

        except Exception as e:
            print(f"Warning: Failed to generate samples: {e}")
            import traceback
            traceback.print_exc()

        self.model.train_mode()

    def save_training_log(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                          grad_norm: Optional[float] = None, max_grad_norm: Optional[float] = None):
        """Save training log to JSON file with gradient information."""
        log_entry = {
            'epoch': epoch + 1,
            'step': self.global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'grad_norm_avg': grad_norm,
            'grad_norm_max': max_grad_norm,
        }

        log_file = self.log_dir / "training_log.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, test_loader: Optional[DataLoader] = None):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (for monitoring during training)
            test_loader: Test data loader (for final evaluation only, NOT used during training)
        """
        print(f"\nStarting training for {self.epochs} epochs")
        print(f"Experiment directory: {self.experiment_dir}\n")

        for epoch in range(self.start_epoch, self.epochs):
            # Train
            train_loss, content_loss, style_loss, tv_loss, grad_norm, max_grad_norm = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['content_loss'].append(content_loss)
            self.history['style_loss'].append(style_loss)
            self.history['tv_loss'].append(tv_loss)
            self.history['epochs'].append(epoch)

            print(f"\n[Epoch {epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f}")
            print(f"  Content: {content_loss:.4f} | Style: {style_loss:.4f} | TV: {tv_loss:.6f}")
            print(f"  Gradient Norm: avg={grad_norm:.4f}, max={max_grad_norm:.4f}")

            # Check for gradient issues
            if grad_norm < 1e-7:
                print(f"  ⚠️  WARNING: Very small gradient norm ({grad_norm:.2e}) - possible vanishing gradients!")
            elif grad_norm > 100:
                print(f"  ⚠️  WARNING: Very large gradient norm ({grad_norm:.2f}) - possible exploding gradients!")

            # Validate (on validation set for monitoring)
            val_loss = None
            if val_loader and (epoch + 1) % self.validate_every == 0:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_epochs'].append(epoch)  # Track validation epoch
                print(f"[Epoch {epoch+1}/{self.epochs}] Val Loss: {val_loss:.4f}")

                # Update scheduler
                if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler:
                    self.scheduler.step()

            # Save training log with gradient info
            self.save_training_log(epoch, train_loss, val_loss, grad_norm, max_grad_norm)

            # Periodic evaluation during training (on TEST set)
            if test_loader and self.evaluate_every and (epoch + 1) % self.evaluate_every == 0:
                print("\n" + "=" * 60)
                print(f"RUNNING PERIODIC EVALUATION AT EPOCH {epoch+1}")
                print("=" * 60)
                self.run_periodic_evaluation(test_loader, epoch)

            # Save training curves every epoch (not just when saving samples/checkpoints)
            plot_training_curves(
                self.history,
                save_path=self.plots_dir / "training_curves.png"
            )
            plot_loss_components(
                self.history,
                save_path=self.plots_dir / "loss_components.png"
            )

            # Generate and save samples (with comparison to real images)
            if (epoch + 1) % self.save_samples_every == 0:
                self.generate_samples(epoch, train_loader)
                print(f"Saved training curves to {self.plots_dir / 'training_curves.png'}")
                print(f"Saved loss components plot to {self.plots_dir / 'loss_components.png'}")

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                is_best = val_loss is not None and val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                state = {
                    'epoch': epoch,
                    'step': self.global_step,
                    'model_state_dict': self.model.mapper.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'config': self.config,
                    'history': self.history,
                    'loss': val_loss if val_loss is not None else train_loss,
                }

                self.ckpt_mgr.save(
                    state,
                    epoch,
                    {'train_loss': train_loss, 'val_loss': val_loss} if val_loss else {'train_loss': train_loss},
                    is_best=is_best
                )

                # Plot curves
                plot_training_curves(
                    self.history,
                    save_path=self.plots_dir / "training_curves.png"
                )

                # Plot loss components
                plot_loss_components(
                    self.history,
                    save_path=self.plots_dir / "loss_components.png"
                )

        print("\nTraining completed!")

        # Run final evaluation on TEST set (not val!)
        if test_loader and self.run_final_evaluation:
            print("\n" + "=" * 60)
            print("IMPORTANT: Running final evaluation on TEST SET (not validation)")
            print("=" * 60)
            self.run_final_evaluation_fn(test_loader, suffix="final")

    def run_periodic_evaluation(self, test_loader: DataLoader, epoch: int):
        """
        Run evaluation during training at periodic intervals.

        Args:
            test_loader: Test data loader
            epoch: Current epoch number
        """
        try:
            from ..evaluation.evaluator import Evaluator

            eval_config = self.config.get("evaluation", {})
            enable_fid = eval_config.get("enable_fid", True)
            enable_is = eval_config.get("enable_is", False)
            enable_lpips = eval_config.get("enable_lpips", False)
            enable_face_semantic = eval_config.get("enable_face_semantic", True)
            max_eval_samples = eval_config.get("max_samples", None)
            num_eval_samples = eval_config.get("num_eval_samples", 16)
            save_images_flag = eval_config.get("save_images", True)

            evaluator = Evaluator(
                model=self.model,
                device=self.device,
                enable_fid=enable_fid,
                enable_is=enable_is,
                enable_lpips=enable_lpips,
                enable_face_semantic=enable_face_semantic,
                num_eval_samples=num_eval_samples,
            )

            # Create periodic evaluation directory structure
            eval_base_dir = self.experiment_dir / "evaluation"
            eval_epochs_dir = eval_base_dir / "evaluation_epochs"
            eval_epochs_dir.mkdir(parents=True, exist_ok=True)

            eval_dir = eval_epochs_dir / f"epoch_{epoch+1:04d}"
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Run evaluation (NO IMAGES - only metrics.json to save memory)
            metrics = evaluator.evaluate(
                dataloader=test_loader,
                save_dir=eval_dir,
                max_samples=max_eval_samples,
                save_images=save_images_flag,
            )

            # Print metrics summary
            print(f"\n📊 Evaluation Results at Epoch {epoch+1}:")
            evaluator.print_metrics(metrics)

            print("=" * 60)
            print(f"Periodic evaluation results saved to {eval_dir}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\nWARNING: Periodic evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            print("Training will continue...\n")

    def run_final_evaluation_fn(self, test_loader: DataLoader, suffix: str = "final"):
        """
        Run comprehensive evaluation after training completes.
        Computes FID, IS, LPIPS, and Face Semantic metrics.

        Args:
            test_loader: Test data loader
            suffix: Suffix for evaluation directory (e.g., "final")
        """
        print("\n" + "=" * 60)
        print(f"RUNNING {'FINAL' if suffix == 'final' else suffix.upper()} EVALUATION")
        print("=" * 60)

        try:
            # Import Evaluator here to avoid circular dependency
            from ..evaluation.evaluator import Evaluator

            # Create evaluator
            eval_config = self.config.get("evaluation", {})
            enable_fid = eval_config.get("enable_fid", True)
            enable_is = eval_config.get("enable_is", False)
            enable_lpips = eval_config.get("enable_lpips", False)
            enable_face_semantic = eval_config.get("enable_face_semantic", True)
            max_eval_samples = eval_config.get("max_samples", None)
            num_eval_samples = eval_config.get("num_eval_samples", 16)
            save_images_flag = eval_config.get("save_images", True)

            evaluator = Evaluator(
                model=self.model,
                device=self.device,
                enable_fid=enable_fid,
                enable_is=enable_is,
                enable_lpips=enable_lpips,
                enable_face_semantic=enable_face_semantic,
                num_eval_samples=num_eval_samples,
            )

            # Create evaluation directory - always use evaluation/evaluation_final
            eval_base_dir = self.experiment_dir / "evaluation"
            eval_dir = eval_base_dir / "evaluation_final"
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Run evaluation
            metrics = evaluator.evaluate(
                dataloader=test_loader,
                save_dir=eval_dir,
                max_samples=max_eval_samples,
                save_images=save_images_flag,
            )

            # Print metrics summary
            evaluator.print_metrics(metrics)

            print("=" * 60)
            print(f"Final evaluation completed! Results saved to {eval_dir}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\nWARNING: Final evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            print("Training results are still saved successfully.\n")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None, load_best: bool = False):
        """Load checkpoint."""
        state = self.ckpt_mgr.load(
            checkpoint_path=checkpoint_path,
            load_best=load_best,
            map_location=str(self.device)
        )

        self.model.mapper.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

        if state.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])

        self.start_epoch = state.get('epoch', 0) + 1
        self.global_step = state.get('step', 0)
        self.history = state.get('history', self.history)

        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
