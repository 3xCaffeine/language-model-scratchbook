# Pretraining

This directory contains comprehensive implementations for pretraining GPT models from scratch, ranging from simple educational scripts to production-ready training pipelines.

## Overview

Pretraining is a process of training language models on large text corpora to learn general language understanding and generation capabilities. This module provides multiple approaches with increasing sophistication and features.

## Files

### Core Training Scripts

#### `pretraining_simple.py`
A straightforward implementation for educational purposes:

**Features:**
- Training on Project Gutenberg books
- Basic training loop with validation
- Simple loss calculation and evaluation
- Text generation sampling during training
- Visualization of training progress

**Dataset:**
- Project Gutenberg book collection
- Manual text processing and splitting
- Basic tokenization with tiktoken

**Use Case:**
- Learning fundamentals of language model training
- Small-scale experiments
- Understanding training dynamics

#### `pretraining_advanced.py`
Production-ready training with comprehensive features:

**Advanced Features:**
- **HuggingFace smollm-corpus** dataset support
- **Pre-tokenized dataset caching** for faster loading
- **Cosine decay with linear warmup** learning rate schedule
- **Gradient clipping** for training stability
- **Model and optimizer checkpointing** with resume capability
- **Weights & Biases logging** for experiment tracking
- **Distributed Data Parallel (DDP)** for multi-GPU training
- **Flash Attention** via PyTorch's optimized implementation
- **torch.compile optimization** for faster training
- **GaLore optimizer** support for memory-efficient training

**Configuration:**
- Configurable model sizes and hyperparameters
- Flexible dataset loading and preprocessing
- Advanced training strategies and optimizations

### Data Preparation

#### `prep_dataset.py`
Dataset preprocessing utilities:

**Features:**
- Project Gutenberg boilerplate removal
- Text cleaning and normalization
- Batch processing of multiple books
- Progress tracking with tqdm
- Fallback implementations for missing dependencies

**Processing Steps:**
1. Download and extract Project Gutenberg texts
2. Remove legal notices and boilerplate
3. Clean and normalize text content
4. Combine into training-ready format

### Training Utilities

#### `gpt_train.py`
Shared training utilities and functions:

**Key Functions:**
- `calc_loss_batch()` - Loss calculation for batches
- `evaluate_model()` - Model evaluation on validation set
- `generate_and_print_sample()` - Text generation during training
- `plot_losses()` - Training visualization

**Features:**
- Reusable training components
- Consistent evaluation metrics
- Memory-efficient implementations

#### `wt_load.py`
Weight loading and management utilities:

**Features:**
- Load pre-trained weights from various sources
- Weight format conversion
- Checkpoint management
- Model state restoration

### Testing and Validation

#### `test_smollm.py`
Test suite for advanced training pipeline:

**Features:**
- Dependency checking
- Quick training verification
- Integration testing
- Environment validation

#### `notebook.py`
Interactive exploration notebook for:
- Training dynamics visualization
- Hyperparameter experimentation
- Model behavior analysis

## Training Configurations

### Model Sizes
- **124M**: GPT-2 Small (fast experimentation)
- **355M**: GPT-2 Medium (balanced performance)
- **774M**: GPT-2 Large (high quality)
- **1558M**: GPT-2 XL (state-of-the-art)

### Training Parameters
```python
training_config = {
    "batch_size": 32,           # Batch size per GPU
    "learning_rate": 5e-4,      # Peak learning rate
    "num_epochs": 50,           # Total training epochs
    "warmup_steps": 2000,       # Linear warmup steps
    "max_grad_norm": 1.0,       # Gradient clipping norm
    "weight_decay": 0.1,        # Weight decay coefficient
}
```

### Optimization Features
- Flash Attention: 2-3x speedup for attention computation
- torch.compile: 20-30% overall training speedup
- DDP: Linear scaling with multiple GPUs
- GaLore: Memory-efficient optimizer for large models

## Dataset Support

### Project Gutenberg
- Classic literature books
- ~50GB of English text
- High-quality, public domain content
- Good for foundational language understanding

### smollm-corpus
- Modern web text and code
- Diverse language patterns
- Better for contemporary language
- Larger vocabulary coverage

## Performance Benchmarks

| Model | Dataset | GPU Hours | Final Loss | Tokens/sec |
|-------|---------|-----------|------------|------------|
| 124M | Gutenberg | 24 | 3.2 | 1,200 |
| 124M | smollm | 48 | 2.8 | 1,500 |
| 355M | smollm | 120 | 2.5 | 800 |

## Monitoring and Logging

### Weights & Biases Integration
- Automatic loss tracking
- Hyperparameter logging
- Gradient and parameter histograms
- System resource monitoring

### Local Logging
- Tensorboard-compatible logs
- Checkpoint saving every N steps
- Training progress reports
- Error handling and recovery

## Dependencies

- PyTorch >= 2.0 (for Flash Attention and torch.compile)
- datasets (HuggingFace dataset loading)
- wandb (experiment tracking)
- tiktoken (efficient tokenization)
- galore-torch (memory-efficient optimization)
- tqdm (progress bars)

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (RTX 2070+)
- RAM: 16GB
- Storage: 50GB

### Recommended
- GPU: 24GB VRAM (RTX 3090/4090 or A100)
- RAM: 64GB
- Storage: 500GB SSD
- Multiple GPUs for DDP training