# Attention Mechanisms

This directory contains implementations and experiments with various attention mechanisms used in modern transformer architectures.

## Overview

Attention mechanisms are the core building blocks of transformer models. This module explores different implementations ranging from basic multi-head attention to optimized variants using PyTorch's built-in functions.

## Files

### `multi_head_attn_impl.py`
Contains three different implementations of multi-head attention:

1. **MultiHeadAttentionCombinedQKV** - A from-scratch implementation that manually handles:
   - Query-Key-Value projection and splitting
   - Causal masking using triangular matrices
   - Attention weight computation and softmax
   - Context vector calculation

2. **MHAPyTorchFlexAttention** - Uses PyTorch's experimental `flex_attention`:
   - Leverages `flex_attention` and `create_block_mask` for efficient computation
   - Supports custom block mask functions (like causal masking)
   - More memory-efficient for large sequences

3. **MHAPyTorchScaledDotProduct** - Uses PyTorch's optimized `scaled_dot_product_attention`:
   - Utilizes Flash Attention when available
   - Handles causal masking automatically
   - Most efficient implementation for production use

### `benchmark.py`
Performance comparison script that benchmarks:
- Execution time across different sequence lengths
- Memory usage patterns
- Numerical accuracy verification between implementations

### `notebook.py`
Interactive exploration notebook for:
- Visualizing attention patterns
- Understanding the mechanics of different attention variants
- Comparing computational graphs

## Key Concepts

### Multi-Head Attention
- Splits the embedding dimension into multiple "heads"
- Each head learns different attention patterns
- Results are concatenated and projected back

### Causal Masking
- Prevents tokens from attending to future positions
- Essential for autoregressive language modeling
- Implemented using triangular masks or built-in functions

### Optimization Techniques
- Flash Attention: Memory-efficient attention computation
- Flex Attention: Customizable attention patterns
- KV Caching: Reuses computed keys and values during generation

## Dependencies

- PyTorch >= 2.0 (for Flash Attention support)
- torch.nn.attention.flex_attention (experimental)