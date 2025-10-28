"""Dataset module for Text-to-Face training."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import torchvision.transforms as transforms


class T2FDataset(Dataset):
    """
    Text-to-Face Dataset.

    Expected structure:
        data_root/
            caps.txt      # TSV file: filename\tcaption
            img/          # Image directory
                image1.jpg
                image2.jpg
                ...

    Args:
        data_root: Root directory containing data
        caps_file: Name of captions file
        img_dir: Name of images directory
        image_size: Target image size
        split: Dataset split ("train", "val", or "test")
        train_split: Train split ratio (default 0.7 for 70%)
        val_split: Val split ratio (default 0.15 for 15%)
        test_split: Test split ratio (default 0.15 for 15%)
        subset_size: Limit dataset size (for testing)
        transform: Custom transform pipeline
        return_paths: Whether to return image paths
        use_augmentation: Enable data augmentation (only for train split)
        augmentation_config: Dict with augmentation parameters
        seed: Optional random seed for deterministic data splits
    """

    def __init__(
        self,
        data_root: str,
        caps_file: str = "caps.txt",
        img_dir: str = "img",
        image_size: int = 256,
        split: str = "train",
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        subset_size: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        return_paths: bool = False,
        use_augmentation: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.img_dir = self.data_root / img_dir
        self.image_size = image_size
        self.split = split
        self.return_paths = return_paths
        self.use_augmentation = use_augmentation and (split == "train")
        self.augmentation_config = augmentation_config or {}
        self.seed = seed

        # Validate split ratios
        if not (0 < train_split < 1 and 0 < val_split < 1 and 0 < test_split < 1):
            raise ValueError("Split ratios must be between 0 and 1")

        total_split = train_split + val_split + test_split
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split:.4f}")

        # Load captions
        caps_path = self.data_root / caps_file
        self.captions, self.image_files = self._load_captions(caps_path)

        # Apply subset limit if specified
        if subset_size is not None:
            self.captions = self.captions[:subset_size]
            self.image_files = self.image_files[:subset_size]

        # Split data into train/val/test
        self._split_data(train_split, val_split, test_split)

        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

    def _load_captions(self, caps_path: Path) -> Tuple[List[str], List[str]]:
        """Load captions from TSV file."""
        if not caps_path.exists():
            raise FileNotFoundError(f"Captions file not found: {caps_path}")

        captions = []
        image_files = []

        with open(caps_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                filename = parts[0]
                caption_text = parts[1]

                img_path = self.img_dir / filename
                if not img_path.exists():
                    continue

                # Handle multi-part captions (separated by |)
                caption_parts = caption_text.split('|')
                full_caption = ' '.join(caption_parts).strip()

                captions.append(full_caption)
                image_files.append(filename)

        return captions, image_files

    def _split_data(self, train_split: float, val_split: float, test_split: float):
        """
        Split data into train/val/test sets with random shuffling.

        Args:
            train_split: Ratio for training set (e.g., 0.7 for 70%)
            val_split: Ratio for validation set (e.g., 0.15 for 15%)
            test_split: Ratio for test set (e.g., 0.15 for 15%)
        """
        import random

        total_samples = len(self.captions)

        # Shuffle data; use configured seed if provided, otherwise fall back to random behavior
        indices = list(range(total_samples))
        if self.seed is not None:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        else:
            random.shuffle(indices)

        # Apply shuffle to both captions and image files
        self.captions = [self.captions[i] for i in indices]
        self.image_files = [self.image_files[i] for i in indices]

        # Calculate split indices
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)
        # test_size is the remainder to handle rounding

        train_end = train_size
        val_end = train_size + val_size

        if self.split == "train":
            self.captions = self.captions[:train_end]
            self.image_files = self.image_files[:train_end]
        elif self.split == "val":
            self.captions = self.captions[train_end:val_end]
            self.image_files = self.image_files[train_end:val_end]
        elif self.split == "test":
            self.captions = self.captions[val_end:]
            self.image_files = self.image_files[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

        seed_info = self.seed if self.seed is not None else "random"
        print(f"Split: {self.split}, Samples: {len(self.captions)}/{total_samples} "
              f"({len(self.captions)/total_samples*100:.1f}%) [Shuffled with seed={seed_info}]")

    def _crop_center_vertical(self, img: Image.Image) -> Image.Image:
        """
        Crop image vertically to make it square, keeping the full width.

        For CelebA images (178x218), this crops 20px from top and 20px from bottom
        to get a square 178x178 image, preserving face centering.

        Args:
            img: PIL Image

        Returns:
            Cropped square PIL Image
        """
        w, h = img.size
        if w == h:
            return img

        # Calculate amount to crop
        crop_amount = h - w  # e.g., 218 - 178 = 40
        top_crop = crop_amount // 2  # 20
        bottom_crop = crop_amount - top_crop  # 20

        # Define crop region (left, top, right, bottom)
        left = 0
        top = top_crop
        right = w
        bottom = h - bottom_crop

        return img.crop((left, top, right, bottom))

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations with vertical center cropping."""
        transform_list = [
            # First crop to square (matching notebook behavior)
            transforms.Lambda(lambda img: self._crop_center_vertical(img)),
            # Then resize to target size
            transforms.Resize((self.image_size, self.image_size)),
        ]

        # Add augmentations for training if enabled
        if self.use_augmentation:
            # Horizontal flip
            if self.augmentation_config.get('horizontal_flip', True):
                flip_prob = self.augmentation_config.get('horizontal_flip_prob', 0.5)
                transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

            # Color jitter
            if self.augmentation_config.get('color_jitter', True):
                brightness = self.augmentation_config.get('brightness', 0.1)
                contrast = self.augmentation_config.get('contrast', 0.1)
                saturation = self.augmentation_config.get('saturation', 0.1)
                hue = self.augmentation_config.get('hue', 0.05)
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue
                    )
                )

            # Random rotation
            if self.augmentation_config.get('rotation', False):
                rotation_degrees = self.augmentation_config.get('rotation_degrees', 10)
                transform_list.append(transforms.RandomRotation(rotation_degrees))

            # Random affine (scale, translate)
            if self.augmentation_config.get('affine', False):
                scale = self.augmentation_config.get('affine_scale', (0.9, 1.1))
                translate = self.augmentation_config.get('affine_translate', (0.1, 0.1))
                transform_list.append(
                    transforms.RandomAffine(
                        degrees=0,
                        translate=translate,
                        scale=scale
                    )
                )

        # Convert to tensor and normalize (always applied)
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        caption = self.captions[idx]
        image_file = self.image_files[idx]
        img_path = self.img_dir / image_file

        # Raise error if image doesn't exist (DataLoader will handle it)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            raise IOError(f"Could not read image {img_path}: {e}")

        sample = {'image': image, 'caption': caption}

        if self.return_paths:
            sample['image_path'] = str(img_path)

        return sample
