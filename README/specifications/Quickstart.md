# Quickstart: Your First Whisper Fine-Tuning

Welcome to the Whisper Fine-Tuner! This guide will walk you through your first successful model fine-tuning in about 30 minutes. We'll use the interactive wizard to fine-tune a small Whisper model on a test dataset - no configuration files or ML expertise required.

## What You'll Accomplish

By the end of this tutorial, you will have:
- ✅ Fine-tuned your first Whisper model using the interactive wizard
- ✅ Understood each step of the training process
- ✅ Generated a model ready for speech recognition tasks
- ✅ Learned how to use your fine-tuned model

**Time Required**: 
- Setup: 5-10 minutes
- Training: ~30 minutes (whisper-tiny on test dataset)
- Total: ~40 minutes

## Prerequisites

Before starting, ensure you have:

1. **Hardware**: 
   - Apple Silicon Mac (M1/M2/M3) with 8GB+ RAM, OR
   - NVIDIA GPU with CUDA support, OR
   - CPU (slower but works)

2. **Software**:
   - Python 3.10 or later
   - Git
   - 5GB free disk space

3. **Basic Knowledge**:
   - How to use a terminal/command line
   - Basic Python environment management (pip/conda)

---

## Step 1: Installation & Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/mattmireles/whisper-fine-tuner-macos.git
cd whisper-fine-tuner-macos
```

### 1.2 Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n whisper-tuner python=3.10
conda activate whisper-tuner
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Verify Your Setup

Run the system check to ensure everything is configured correctly:

```bash
python scripts/system_check.py
```

You should see output similar to:
```
✅ Python 3.10.12 (arm64)
✅ PyTorch 2.0.1 with MPS support
✅ Apple Silicon detected: M2 Pro
✅ Available memory: 16.0 GB
✅ MPS (Metal Performance Shaders) is available
```

---

## Step 2: Launch the Interactive Wizard

The wizard provides a Steve Jobs-inspired interface that guides you through the entire process with smart defaults for everything.

```bash
python wizard.py
```

You'll see a welcome screen:

```
██╗    ██╗██╗  ██╗██╗███████╗██████╗ ███████╗██████╗ 
██║    ██║██║  ██║██║██╔════╝██╔══██╗██╔════╝██╔══██╗
██║ █╗ ██║███████║██║███████╗██████╔╝█████╗  ██████╔╝
██║███╗██║██╔══██║██║╚════██║██╔═══╝ ██╔══╝  ██╔══██╗
╚███╔███╔╝██║  ██║██║███████║██║     ███████╗██║  ██║
 ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝

Welcome to the Whisper Fine-Tuning Wizard!
System Status: ✅ Apple Silicon (MPS) | 16GB RAM | Ready
```

---

## Step 3: Wizard Walkthrough (Now with Gemma)

The wizard will ask you 6 simple questions. Here's exactly what you'll see and the recommended answers for your first training:

### Question 0: Model Family

```
? Choose your model family:
  ❯ 🌬️ Whisper - OpenAI's robust ASR model
    💎 Gemma - Google's new multimodal model (audio+text)
```

If you choose Gemma, the wizard restricts training to **LoRA** and filters the model list to Gemma variants. The confirmation screen also surfaces Gemma dtype (bf16 preferred on MPS) and attention implementation (eager).

### Question 1: Training Method

```
? Choose your training method:
  ❯ 🚀 Standard Fine-Tune (SFT) - Highest quality
    🎨 LoRA Fine-Tune - Memory efficient (40% less RAM)
    🧠 Knowledge Distillation - Create smaller, faster models
```

**Recommended Answer**: Select "🚀 Standard Fine-Tune (SFT)"

**Why**: Standard fine-tuning provides the best accuracy and is perfect for learning the system.

### Question 2: Model Selection

```
? Which model do you want to fine-tune?
  ❯ whisper-tiny (39M) - ~30 min, 1.2GB memory ⭐ Recommended
    whisper-base (74M) - ~1 hour, 2.1GB memory
    whisper-small (244M) - ~2.5 hours, 4.2GB memory
    whisper-medium (769M) - ~6 hours, 8.4GB memory
```

**Recommended Answer**: Select "whisper-tiny"

**Why**: Tiny is the fastest to train and perfect for testing. You can move to larger models once familiar with the process.

### Question 3: Dataset Selection

```
? Which dataset do you want to use for training?
  ❯ 📁 test_streaming - Local dataset with 2 samples
    🤗 Browse HuggingFace datasets...
    📊 Import from BigQuery
    🗂️ Browse for custom dataset...
```

**Recommended Answer**: Select "📁 test_streaming"

**Why**: This is a minimal test dataset included with the repository, perfect for verification.

### Question 4: Learning Rate

```
? Learning rate (default: 1e-5): 
```

**Recommended Answer**: Press Enter to accept default (1e-5)

**Why**: The default learning rate is well-tested for Whisper fine-tuning.

### Question 5: Number of Epochs

```
? Number of training epochs (default: 3): 
```

**Recommended Answer**: Press Enter to accept default (3)

**Why**: 3 epochs is standard for fine-tuning and prevents overfitting on small datasets.

### Question 6: Warmup Steps

```
? Number of warmup steps (default: 500): 50
```

**Recommended Answer**: Type "50" and press Enter

**Why**: For tiny datasets, fewer warmup steps are appropriate.

### Confirmation Screen

The wizard will show a summary:

```
┌─────────────────────────────────────┐
│ Training Configuration              │
├─────────────────────────────────────┤
│ Method:     🚀 Standard Fine-Tune   │
│ Model:      whisper-tiny            │
│ Dataset:    test_streaming (2)      │
│ Learning:   1e-5                    │
│ Epochs:     3                       │
│ Warmup:     50                      │
│                                     │
│ Estimated:  30 minutes              │
│ Memory:     1.2 GB                  │
│ Device:     Apple Silicon (mps)     │
└─────────────────────────────────────┘

? Ready to start training? (y/n): y
```

**Answer**: Type "y" and press Enter to begin training!

---

## Step 4: Training Process

Once you confirm, the training begins automatically. You'll see progress updates:

```
Starting training run 48-wizard_20250115_143022...
Loading whisper-tiny model...
✅ Model loaded successfully
Processing dataset...
✅ Dataset ready: 2 training samples

Training Progress:
[████████████████████████████████████] 100% | Epoch 3/3 | Loss: 0.234

✅ Training completed successfully!
📁 Output directory: output/48-wizard_20250115_143022/
```

### What's Happening Behind the Scenes

1. **Model Loading**: Downloads the base whisper-tiny model from HuggingFace
2. **Dataset Processing**: Loads audio files and transcriptions
3. **Training Loop**: Updates model weights based on your data
4. **Checkpoint Saving**: Saves progress periodically
5. **Export**: Optionally converts to GGUF format for whisper.cpp

---

## Step 5: Understanding Your Output

After training completes, your fine-tuned model is saved in the output directory:

```
output/48-wizard_20250115_143022/
├── model.safetensors          # PyTorch model weights
├── config.json                # Model configuration
├── tokenizer_config.json      # Tokenizer settings
├── vocab.json                 # Vocabulary
├── merges.txt                # BPE merges
├── metadata.json             # Training metadata
├── metrics.json              # Training metrics
├── run.log                   # Detailed training log
└── completed                 # Success marker file
```

### Key Files Explained

- **`model.safetensors`**: Your fine-tuned model weights. This is what you'll use for inference.
- **`config.json`**: Model architecture configuration
- **`metadata.json`**: Complete record of training parameters and settings
- **`run.log`**: Detailed log useful for debugging

### Optional GGUF Export

If whisper.cpp is installed, you may also see:
- **`model-f16.gguf`**: Optimized format for CPU inference with whisper.cpp

---

## Step 6: Testing Your Model

### Quick Test with Python

Create a test script `test_model.py`:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load your fine-tuned model
model_path = "output/48-wizard_20250115_143022"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Load and transcribe audio
import librosa
audio, sr = librosa.load("data/datasets/test_streaming/audio/tiny.wav", sr=16000)

# Process and generate transcription
inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
generated_ids = model.generate(inputs.input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Transcription: {transcription}")
```

### Using with whisper.cpp (if GGUF was exported)

```bash
# If you have whisper.cpp installed
./whisper.cpp/main -m output/48-wizard_20250115_143022/model-f16.gguf -f audio.wav
```

---

## Step 7: Next Steps

Congratulations! You've successfully fine-tuned your first Whisper model. Here's what to explore next:

### Use Your Own Data

1. Prepare a CSV file with columns: `id`, `audio_path`, `text`
2. Place audio files in an accessible location
3. Run the wizard and select "Browse for custom dataset"

Example CSV format:
```csv
id,audio_path,text
1,/path/to/audio1.wav,"Hello world"
2,/path/to/audio2.mp3,"Welcome to Whisper fine-tuning"
```

### Try Advanced Methods

- **LoRA Fine-Tuning**: Uses 60% less memory, great for larger models
- **Knowledge Distillation**: Create smaller, faster models from large teachers

### Optimize for Production

- Use larger models (small, medium) for better accuracy
- Train for more epochs with larger datasets
- Experiment with learning rates and batch sizes

### Learn More

- [SFT Specification](../specifications/SFT.md): Deep dive into standard fine-tuning
- [LoRA Specification](../specifications/LoRA.md): Memory-efficient training
- [Distillation Specification](../specifications/Distillation.md): Model compression
- [Wizard Specification](../specifications/wizard.md): Complete wizard documentation

---

## Troubleshooting

### Common Issues and Solutions

**Issue**: "CUDA out of memory" or "MPS ran out of memory"
- **Solution**: Use LoRA instead of standard fine-tuning, or choose a smaller model

**Issue**: Training is very slow
- **Solution**: Ensure you're using GPU (MPS/CUDA), not CPU. Check with `python scripts/system_check.py`

**Issue**: Model not improving (loss not decreasing)
- **Solution**: Try a higher learning rate (5e-5 or 1e-4) or more training epochs

**Issue**: Can't find the wizard
- **Solution**: Make sure you're in the repository root directory and run `python wizard.py`

### Getting Help

- Check the [Known Issues](../KNOWN_ISSUES.md) document
- Review the comprehensive [specifications](../specifications/) for detailed information
- Open an issue on GitHub with your `run.log` file for debugging help

---

## Summary

You've learned how to:
- ✅ Set up the Whisper Fine-Tuner on your system
- ✅ Use the interactive wizard for configuration-free training
- ✅ Fine-tune a Whisper model on a test dataset
- ✅ Understand the output files and their purposes
- ✅ Test your fine-tuned model

The wizard makes fine-tuning accessible to everyone, hiding complexity while maintaining professional-grade capabilities. As you become more comfortable, you can explore larger models, custom datasets, and advanced training methods.

Happy fine-tuning! 🎉