# GPT Model Implementations

This directory contains various implementations of GPT (Generative Pre-trained Transformer) models, including standard architectures and experimental variants with different attention mechanisms.

## Overview

The GPT model module provides complete implementations of transformer-based language models, from basic GPT-2 architecture to advanced variants with optimized attention mechanisms. All implementations support training, inference, and weight loading from pre-trained models.

## Files

### Core Implementations

#### `gpt.py`
Standard GPT-2 implementation with all essential components:

**Key Classes:**
- `MultiHeadAttention` - Standard multi-head attention with KV caching
- `LayerNorm` - Layer normalization with learnable parameters
- `GELU` - GELU activation function
- `FeedForward` - Position-wise feed-forward network
- `TransformerBlock` - Complete transformer block with attention and FFN
- `GPTModel` - Full GPT model with embedding, transformer blocks, and output head

**Features:**
- KV caching for efficient generation
- Support for different model sizes (124M, 355M, 774M, 1558M)
- Pre-trained weight loading from Hugging Face
- Text generation utilities

### Experimental Variants

#### `gpt_gqa.py`
GPT with **Grouped Query Attention (GQA)** for memory efficiency:

**Key Features:**
- Reduced memory usage by sharing key/value heads across multiple query heads
- Configurable number of KV groups (`num_kv_groups`)
- Maintains model quality while reducing computational cost
- KV caching support

**Benefits:**
- 2-4x reduction in KV cache memory
- Faster inference for long sequences
- Minimal quality degradation

#### `gpt_mla.py`
GPT with **Multi-Head Latent Attention (MLA)** inspired by DeepSeek:

**Key Features:**
- Projects keys/values to lower-dimensional latent space
- Reduces memory footprint and computational cost
- Configurable latent dimension
- Latent KV caching

**Benefits:**
- Significant memory savings for large models
- Faster attention computation
- Maintains expressiveness through latent representations

#### `gpt_swa.py`
GPT with **Sliding Window Attention (SWA)** for efficient long-sequence processing:

**Key Features:**
- Local attention within sliding windows
- Linear complexity with sequence length
- Configurable window size
- Maintains causality

**Benefits:**
- O(n) complexity instead of O(n²)
- Efficient for very long sequences
- Good performance on document-level tasks

### Utilities

#### `load_weights.py`
Weight loading utilities for GPT models:

**Features:**
- Load pre-trained weights from Hugging Face
- Cache weights locally for faster loading
- Support for all GPT-2 model sizes
- PyTorch compatibility layer

#### `dataset_loader.py`
Data loading utilities for training:

**Classes:**
- `GPTDatasetV1` - Dataset class with sliding window chunking
- `create_dataloader_v1` - DataLoader factory function

**Features:**
- Efficient text tokenization with tiktoken
- Sliding window for overlapping sequences
- Configurable batch size and sequence length
- Support for shuffling and multi-worker loading

#### `test_weight_loading.py`
Test suite for weight loading functionality.

#### `perf-analysis.py`
Performance analysis tools for model benchmarking.

#### `notebook.py`
Interactive exploration notebook for model analysis.

## Model Configurations

### Standard GPT-2 Sizes
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

### Experimental Variants
- GQA: `num_kv_groups` parameter (typically 2-4)
- MLA: `latent_dim` parameter (typically d_out // 8)
- SWA: `sliding_window_size` parameter (typically 256-1024)

## Performance Characteristics

| Variant | Memory Usage | Speed | Quality | Best For |
|---------|-------------|-------|---------|----------|
| Standard | Baseline | Baseline | Best | General purpose |
| GQA | 2-4x less | Faster | Slightly lower | Long sequences |
| MLA | 3-5x less | Much faster | Good | Large models |
| SWA | Linear | Much faster | Good | Very long documents |

## Dependencies

- PyTorch >= 2.0
- tiktoken for tokenization
- transformers for weight loading
- numpy for numerical operations

## Advanced Features

### KV Caching
All implementations support KV caching for efficient autoregressive generation:
```python
model.reset_kv_cache()  # Reset cache
output = model(input_ids, use_cache=True)  # Use cache
```

### Flash Attention
Uses PyTorch's `scaled_dot_product_attention` for optimal performance when available.

### Distributed Training
Models can be wrapped with `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel` for multi-GPU training.