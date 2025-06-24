# Whisper Fine-Tuner for macOS

A comprehensive framework for fine-tuning OpenAI's Whisper models with native Apple Silicon support via Metal Performance Shaders (MPS).

## Features

- 🚀 **Native Apple Silicon Support**: Optimized for M1/M2/M3 chips using MPS
- 🔄 **Cross-Platform**: Also supports NVIDIA GPUs (CUDA) and CPU
- 🎯 **Multiple Model Support**: Whisper (small, medium, large-v2) and Distil-Whisper
- 📊 **Comprehensive Evaluation**: WER/CER metrics with detailed analysis
- 🔍 **Outlier Detection**: Automatic blacklisting of problematic samples
- 🏷️ **Pseudo-Labeling**: Generate labels for unlabeled data
- 📦 **Export to GGML**: Convert models for whisper.cpp

## System Requirements

### For Apple Silicon (Recommended)
- **macOS**: 12.3+ (Monterey or later)
- **Hardware**: Apple Silicon Mac (M1/M2/M3)
- **Python**: 3.8+ (ARM64 native - NOT x86_64/Rosetta)
- **RAM**: 16GB minimum, 32GB+ recommended

### For NVIDIA GPUs
- **CUDA**: 11.7+
- **GPU**: NVIDIA GPU with 8GB+ VRAM

## Installation

### 1. Verify ARM64 Python (Apple Silicon only)
```bash
python -c "import platform; print(platform.machine())"
# Should output: arm64
# If it shows x86_64, you're using Rosetta - reinstall Python/Conda for ARM64
```

### 2. Install PyTorch with MPS support
```bash
# For Apple Silicon
pip install torch torchvision torchaudio

# For CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies
```bash
pip install transformers datasets evaluate librosa soundfile accelerate
pip install packaging filelock
```

### 4. Verify MPS setup
```bash
python scripts/system_check.py
```

## Quick Start

### 1. Prepare your dataset
Create a CSV file with columns:
- `audio_path`: Path to audio files
- `text_perfect`: Target transcription
- `note_id`: Unique identifier

### 2. Configure training
Edit `config.ini` to set:
- Model size (small/medium/large)
- Batch sizes (start with defaults)
- Training parameters

### 3. Run fine-tuning
```bash
# For Apple Silicon - enable fallback for initial testing
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run training
python main.py --profile medium-data3
```

### 4. Evaluate model
```bash
python scripts/evaluate.py --model_name_or_path output/run-001-medium-data3 --dataset data3
```

## Apple Silicon Optimization

### Environment Variables
```bash
# Enable CPU fallback for unsupported operations (debugging only)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set memory limit (0.8 = 80% of system RAM)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
```

### Recommended Batch Sizes
| Model | M1/M2 (16-24GB) | M1/M2 Max (32-64GB) | M1/M2 Ultra (64-192GB) |
|-------|-----------------|---------------------|------------------------|
| Small | 16 | 24 | 32+ |
| Medium | 8 | 16 | 24+ |
| Large-v2 | 4 | 8 | 16+ |

### Performance Tips
1. **Start small**: Use conservative batch sizes initially
2. **Monitor memory**: Use Activity Monitor to check memory pressure
3. **Gradual increase**: Increase batch sizes if no swapping occurs
4. **Gradient accumulation**: Use to simulate larger batches

## Project Structure
```
whisper-fine-tuner-macos/
├── models/
│   ├── whisper/         # Standard Whisper models
│   └── distil-whisper/  # Distilled Whisper models
├── scripts/
│   ├── system_check.py  # Verify GPU/MPS setup
│   ├── evaluate.py      # Model evaluation
│   ├── blacklist.py     # Outlier detection
│   └── export.py        # GGML conversion
├── utils/
│   └── device.py        # Device selection (MPS/CUDA/CPU)
├── config.ini           # Training configurations
└── main.py             # Main entry point
```

## Common Issues

### MPS-Specific Issues
1. **"PyTorch not compiled with MPS"**: Reinstall PyTorch (ensure ARM64 Python)
2. **Memory errors**: Reduce batch size or set PYTORCH_MPS_HIGH_WATERMARK_RATIO
3. **Slow performance**: Disable PYTORCH_ENABLE_MPS_FALLBACK after testing

### General Issues
1. **Import errors**: Check all dependencies are installed
2. **OOM errors**: Reduce batch size or enable gradient checkpointing
3. **Data loading**: Ensure audio files are accessible and valid

## Advanced Usage

### Multi-GPU Training (CUDA only)
```bash
torchrun --nproc_per_node=2 main.py --profile large-v2-data3
```

### Custom Configurations
Create new profiles in `config.ini`:
```ini
[profile:custom-profile]
model = whisper-medium
dataset = your-dataset
learning_rate = 1e-5
per_device_train_batch_size = 12
```

### Export to CoreML (coming soon)
For maximum inference performance on Apple Silicon, export to CoreML after training.

## Acknowledgments

- OpenAI for Whisper
- Hugging Face for Transformers
- PyTorch team for MPS support