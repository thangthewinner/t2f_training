# Text-to-Face (T2F) Generation with StyleGAN2

[![Python 3.12.11](https://img.shields.io/badge/python-3.12.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-ready PyTorch implementation of **Text-to-Face Generation with StyleGAN2** by Ayanthi & Munasinghe (2022). Generate high-quality face images (1024×1024) from textual descriptions using BERT embeddings and StyleGAN2.

---

## Key Features

- **High Quality**: 1024×1024 face generation using StyleGAN2 pre-trained on FFHQ
- **Flexible**: Support for Z, W, and W+ latent spaces (W+ recommended)
- **Production Ready**: Modular codebase, comprehensive logging, checkpoint management
- **Face Semantic Metrics**: FaceNet-based evaluation (matching paper implementation)
- **Easy Resume**: Resume training from checkpoint without config file
- **Smart Inference**: Generate images directly from checkpoint (no config needed)

---

## Architecture

```
Text Description → BERT Encoder → Mapper Network → StyleGAN2 Generator → Face Image
                    (frozen)        (trainable)         (frozen)
```

### Pipeline Details

1. **Text Encoder (BERT)**: `bert-base-uncased` with mean pooling → 768D embeddings (frozen, runs on CPU)
2. **Mapper Network**: MLP 768 → [intermediate] → 9216 (trainable)
3. **StyleGAN2**: Pre-trained FFHQ 1024×1024 generator (frozen)
4. **Perceptual Loss**: VGG16-based content loss + style loss + TV loss (total variation regularization)

## Quick Start

### Setup Environment

**Option 1: Using Python venv (Recommended)**
```bash
# Clone repo
git clone https://github.com/thangthewinner/t2f_training
cd t2f_training

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CUDA 12.1)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Using uv (Faster)**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

### Download Pretrained Models

**StyleGAN2 (Required):**
```bash
# Clone StyleGAN2 repository
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

# Download FFHQ pretrained weights (required for training)
mkdir -p pretrained_model
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
     -O pretrained_model/ffhq.pkl
```

**Trained T2F Model (Optional - for inference only):**

Pre-trained T2F mapper after 500 epochs on W+ space (ready to use):
- **Download**: [T2F W+ Model (500 epochs)](https://drive.google.com/file/d/1FHynclzxW_KTxkekz1pMGTUludYIkKhv/view?usp=sharing)
- **Config**: W+ latent space (`wplus_space.yaml`)
- **Dataset**: 6000 CelebA images with Text2FaceGAN captions
- **Usage**: Download and use directly with `generate.py` for inference

### Dataset Preparation

This project uses the **Text2FaceGAN dataset** - automatically generated captions for CelebA images based on 40 facial attributes.

#### Step 1: Download Full Dataset

**1. Download CelebA Images:**
- Source: [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- Download `img_align_celeba.zip` (1.3GB, 202,599 images)
- Extract to: `origin_data/img/img_align_celeba/`

**2. Download Text2FaceGAN Captions:**
- Source: [Text2FaceGAN Repository](https://github.com/midas-research/text2facegan)
- Download `caps.txt` from the repository
- Place in: `origin_data/caps.txt`

Your directory structure should look like:
```
origin_data/
├── caps.txt                           # Captions file
└── img/
    └── img_align_celeba/              # CelebA images
        ├── 000001.jpg
        ├── 000002.jpg
        └── ... (202,599 images)
```

#### Step 2: Reduce Dataset (Recommended)

For faster training and experimentation, reduce the full dataset to 6000 samples:

```bash
# Reduce to 6000 samples (default) from full dataset
python reduce_data.py
```

**Final Dataset Structure:**
```
data/
├── caps.txt          # Subset captions (6000 samples)
└── img/              # Subset images
    ├── 000001.jpg
    ├── 000002.jpg
    └── ... (6000 images)
```

**Example `caps.txt` format:**
```tsv
000001.jpg	The woman has high cheekbones.|She has straight hair which is brown in colour.|She has pointy nose with arched eyebrows and a slightly open mouth.|The smiling, young attractive woman has heavy makeup.|She's wearing earrings and lipstick.
000002.jpg	The woman has high cheekbones.|Her hair is brown in colour.|She has big nose and a slightly open mouth.|The young woman is smiling.
```

**Customize Reduction (Optional):**

You can modify the parameters in `reduce_data.py`:
```python
reduce_dataset(
    caps_file="./origin_data/caps.txt",                    # Source captions
    target_caps_file="./data/caps.txt",                    # Target captions
    img_source_dir="./origin_data/img/img_align_celeba",   # Source images
    img_target_dir="./data/img",                           # Target images
    subset_size=6000,                                      # Number of samples (default: 6000)
)
```

---

## Training

### Fresh Training

```bash
# Quick test (100 samples, 10 epochs) - 4GB GPU
python scripts/train.py --config configs/wplus_space_quick_test_4gb.yaml

# Full training - W+ space (recommended for best quality)
python scripts/train.py --config configs/wplus_space.yaml

# W space (faster, less expressive)
python scripts/train.py --config configs/w_space.yaml

# Z space (fastest, least expressive)
python scripts/train.py --config configs/z_space.yaml
```

### Resume Training

**Method 1: No config needed (Recommended)**

```bash
# Simple resume - automatically loads all settings from checkpoint
python scripts/train.py --resume-from checkpoint_epoch0050.pt

# Resume with extended epochs
python scripts/train.py \
  --resume-from checkpoint_epoch0050.pt \
  --override training.epochs=200

# Resume with adjusted learning rate
python scripts/train.py \
  --resume-from checkpoint_epoch0050.pt \
  --override training.optimizer.lr=5e-5

# Multiple overrides
python scripts/train.py \
  --resume-from checkpoint.pt \
  --override \
    training.epochs=200 \
    training.optimizer.lr=5e-5 \
    training.batch_size=2
```

**Safe override parameters:**
- `training.epochs` - Extend training duration
- `training.optimizer.lr` - Adjust learning rate
- `training.batch_size` - Change batch size
- `training.gradient_accumulation_steps` - Adjust gradient accumulation
- `experiment.device` - Switch device (cuda/cpu)
- `experiment.seed` - Change random seed

**Note:** Cannot override model architecture (e.g., `model.latent_space`) - this would break checkpoint compatibility.

**Method 2: With config file (Legacy)**

```bash
# Resume from specific checkpoint
python scripts/train.py \
  --config configs/wplus_space.yaml \
  --resume experiments/t2f_wplus_20231115/checkpoints/checkpoint_epoch0050.pt

# Resume from best checkpoint
python scripts/train.py \
  --config configs/wplus_space.yaml \
  --resume-best
```

### Monitor Training

```bash
# View logs in real-time
tail -f experiments/your_experiment/logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check generated samples
ls experiments/your_experiment/samples/
```

---

## Inference / Generation 

#### Simple Command-Line Inference (No config needed!)
```bash
# Generate from checkpoint directly
python scripts/generate.py \
  --checkpoint checkpoint.pt \
  --captions "A young woman with blonde hair" "A man with short brown hair" \
  --output-dir outputs/

# Batch generation from file
echo "A young woman with blonde hair" > prompts.txt
echo "A man with short brown hair" >> prompts.txt

python scripts/generate.py \
  --checkpoint checkpoint.pt \
  --captions-file prompts.txt \
  --output-dir outputs/

# Interpolation between two captions
python scripts/generate.py \
  --checkpoint checkpoint.pt \
  --interpolate "A young woman" "An elderly man" \
  --interpolate-steps 10 \
  --output-dir interpolation/
```

#### Programmatic Usage
```python
from src.models.t2f_model import T2FModel
from src.utils.checkpoint_utils import load_config_from_checkpoint
import torch
from pathlib import Path

# Load config from checkpoint
checkpoint_path = Path("checkpoint.pt")
config = load_config_from_checkpoint(checkpoint_path)

# Create model
model = T2FModel(config=config)

# Load weights
ckpt = torch.load(checkpoint_path, map_location='cuda')
model.mapper.load_state_dict(ckpt['model_state_dict'])
model.eval_mode()

# Generate
captions = ["A young woman with blonde hair", "A man with beard"]
with torch.no_grad():
    images = model(captions)  # (B, 3, 1024, 1024), range [-1, 1]
```

**Output:** Images are saved to the specified directory with filenames like `generated_0001.png`, `generated_0002.png`.

---

## Project Structure

```
t2f_training/
├── src/                           # Modular source code
│   ├── models/                    # BERT, Mapper, StyleGAN2, T2F
│   ├── losses/                    # Perceptual loss (VGG16)
│   ├── data/                      # Dataset with crop_center_vertical
│   ├── training/                  # Trainer, optimizer, checkpoints
│   ├── evaluation/                # FID, IS, LPIPS, FSD/FSS (FaceNet)
│   └── utils/                     # Logging, memory, visualization
├── configs/                       # YAML configurations
│   ├── wplus_space.yaml           # Recommended (W+ space)
│   ├── wplus_space_quick_test_4gb.yaml  # Quick test config
│   ├── w_space.yaml               # W space
│   └── z_space.yaml               # Z space
├── scripts/
│   ├── train.py                   # Main training script
│   └── generate.py                # Generation/inference script
├── origin_data/                   # Original full dataset
│   ├── caps.txt                   # Full captions (202,599)
│   └── img/
│       └── img_align_celeba/      # Full CelebA images
├── data/                          # Reduced dataset (for training)
│   ├── caps.txt                   # Subset captions (6000)
│   └── img/                       # Subset images (6000)
├── pretrained_model/              # StyleGAN2 checkpoint
│   └── ffhq.pkl                   # FFHQ pretrained weights
├── stylegan2-ada-pytorch/         # NVIDIA's StyleGAN2 repo
├── experiments/                   # Training outputs (auto-created)
├── reduce_data.py                 # Dataset reduction script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Citation

**Original Paper:**
```bibtex
@article{ayanthi2022text2face,
  title={Text-to-Face Generation with StyleGAN2},
  author={D. M. A. Ayanthi and Munasinghe, Sarasi},
  journal={arXiv preprint arXiv:2205.12512},
  year={2022},
  url={https://arxiv.org/abs/2205.12512}
}
```
---

## License

MIT License - see [LICENSE](LICENSE) for details.
