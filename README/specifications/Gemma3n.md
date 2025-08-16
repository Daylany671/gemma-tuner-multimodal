# Gemma 3n Multimodal Fine-Tuning Product Specification

## Executive Summary

This document outlines the integration of Google's Gemma 3n, a state-of-the-art open multimodal model, into the Whisper Fine-Tuner framework. This extension enables users to perform parameter-efficient fine-tuning (PEFT) on Gemma 3n's audio capabilities, leveraging the framework's existing Apple Silicon (MPS) optimizations. The core of this project involves engineering a robust data pipeline to accommodate Gemma's unique audio processing requirements and extending the CLI wizard to provide a seamless, guided user experience for this new model family.

### Key Capabilities

- **Gemma 3n Audio Fine-Tuning**: Adapt Gemma 3n for domain-specific audio transcription tasks.
- **Parameter-Efficient Training**: Utilize LoRA and QLoRA to fine-tune Gemma 3n on consumer hardware.
- **Seamless Wizard Integration**: A guided, zero-configuration workflow for setting up Gemma 3n training runs.
- **Apple Silicon First**: Optimized for PyTorch on MPS, leveraging unified memory for efficient training.
- **Multimodal Data Pipeline**: A new data processor to handle the Universal Speech Model (USM) feature extraction required by Gemma 3n.

## Technical Architecture

### System Integration

Gemma 3n will be integrated as a new model family alongside Whisper. The existing architecture (`core/ops.py` dispatch, `models/*/finetune.py` structure) will be extended.

- **Model Implementation**: A new directory `models/gemma/finetune.py` will be created. It will leverage Hugging Face's `transformers` library to load `AutoModelForCausalLM` and `AutoProcessor` for Gemma 3n models.
- **Training Framework**: The project will use the `trl.SFTTrainer`, which is well-suited for Gemma's chat-based format. This requires a specialized data pipeline to format audio-text pairs into the required conversational structure.
- **Primary Toolkit**: **PyTorch with MPS** is the designated framework for this task, as recommended in the developer field guide for its mature ecosystem (`peft`, `trl`) and flexibility with complex multimodal models. MLX is explicitly avoided due to its current instability with Gemma's audio tower.

### Core Technical Challenges & Solutions

1.  **Data Preprocessing (The Critical Path)**:
    - **Challenge**: Gemma's audio encoder uses Google's Universal Speech Model (USM), which requires a specific feature extraction process, unlike Whisper's simple log-mel spectrogram.
    - **Solution**: We will create a new data preparation script, `utils/gemma_dataset_prep.py`. This script will use the official `transformers.GemmaProcessor` to handle all audio processing. This ensures perfect replication of the required feature extraction and tokenization.

2.  **Conversational Data Formatting**:
    - **Challenge**: `SFTTrainer` for Gemma requires input data in a specific chat format with special tokens (`<bos>`, `<start_of_turn>`, `<end_of_turn>`).
    - **Solution**: The new data prep script will include a function to transform simple `(audio, text)` pairs into the required JSONL format with the correct conversational structure. The `<bos>` token will be prepended to every training example, a strict requirement for stable training.

3.  **Numerical Stability on MPS**:
    - **Challenge**: Gemma was pre-trained using `bfloat16`. The PyTorch MPS backend can be sensitive to floating-point precision, potentially leading to `NaN` loss values when using the default `float16`.
    - **Solution**: The training script (`models/gemma/finetune.py`) and wizard will default to using `bfloat16` (`bf16=True` in `SFTConfig`) when an MPS device is detected. If the hardware does not support `bfloat16`, it will fall back to full `float32`, and the user will be warned about increased memory usage.

## CLI Wizard Integration (`wizard.py`)

The wizard will be extended to make Gemma 3n a first-class citizen, maintaining the principle of progressive disclosure.

### New Wizard Flow

1.  **Top-Level Model Family Selection (New Step)**:
    The first choice after the welcome screen will be the model family.
    ```
    ? Choose the model family you want to work with:
      ❯ 🌬️ Whisper - The robust ASR model from OpenAI.
        💎 Gemma - The new multimodal model from Google.
    ```

2.  **Gemma Model Selection (Conditional Step)**:
    If the user selects "Gemma", they will see a Gemma-specific model selection screen.
    ```
    ? Which Gemma 3n model do you want to fine-tune?
      ❯ gemma-3n-E2B-it (Elastic 2B) - Faster, smaller memory footprint. ⭐ Recommended
        gemma-3n-E4B-it (Elastic 4B) - Maximum capability, higher memory usage.
    ```
    The wizard will perform a memory check and hide options that are not feasible on the user's hardware.

3.  **Training Method Adaptation**:
    The training method screen will be adapted. Initially, only LoRA will be offered for Gemma due to the high memory requirements of a full SFT.
    ```
    ? Choose your training method for Gemma:
      ❯ 🎨 LoRA Fine-Tune - The only way to fly on consumer hardware.
    ```

4.  **Automatic Configuration**:
    The wizard will handle all Gemma-specific configuration **automatically** behind the scenes:
    - It will generate a training configuration that uses `bfloat16` on MPS devices.
    - It will set up the data pipeline to use the new `gemma_dataset_prep.py` script.
    - It will configure the `SFTTrainer` with the correct chat template and ensure the `<bos>` token is prefixed.
    - The user will not be burdened with these details.

5.  **Confirmation Screen Update**:
    The confirmation screen will clearly state the chosen model is Gemma and show relevant parameters.
    ```
    ┌─────────────────────────────────────┐
    │ Training Configuration              │
    ├─────────────────────────────────────┤
    │ Family:     💎 Gemma                 │
    │ Model:      gemma-3n-E2B-it         │
    │ Method:     🎨 LoRA Fine-Tune       │
    │ Dataset:    common_voice (50k)      │
    ...
    └─────────────────────────────────────┘
    ```

## Configuration System (`config.ini`)

To support Gemma, the configuration system will be extended with a new group and model profiles.

### New `[group:gemma]` Section

```ini
[group:gemma]
# Common settings for all Gemma models
attn_implementation = eager
dtype = bfloat16 ; Critical for MPS stability
optim = paged_adamw_32bit
```

### New Model/Profile Sections

```ini
[model:gemma-3n-e2b-it]
base_model = google/gemma-3n-E2B-it
group = gemma

[profile:gemma-lora-test]
inherits = DEFAULT
model = gemma-3n-e2b-it
dataset = test_streaming
method = lora
lora_r = 16
lora_alpha = 32
target_modules = q_proj,k_proj,v_proj,o_proj
```

## Limitations and Challenges

- **High Memory Requirements**: Even with LoRA, fine-tuning Gemma 3n's audio tower is memory-intensive. Full SFT will be impractical on most consumer hardware.
- **Data Pipeline Complexity**: The dependency on the `GemmaProcessor` and the specific chat template makes the data pipeline more fragile than Whisper's. Any deviation will lead to poor results.
- **Initial Scope**: The initial integration will focus on LoRA fine-tuning for audio transcription. Other modalities (vision) and training methods (distillation) are out of scope for the first version.
- **MLX Instability**: As noted in the field guide, `mlx-lm` has known issues with Gemma's audio tower. This integration will **only** support the PyTorch MPS backend.

## Implementation Progress 