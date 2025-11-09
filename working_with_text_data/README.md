# Working with Text Data

This directory contains comprehensive implementations and experiments with text processing, tokenization, and byte-pair encoding (BPE) algorithms. It includes both from-scratch implementations and performance comparisons with official tokenizers.

## Overview

Text processing is the foundation of language modeling. This module explores the complete pipeline from raw text to tokenized sequences, including custom BPE implementations, regex-based splitting, and benchmarking against industry-standard tokenizers.

## Directory Structure

### `bytepair-enc/`
Complete byte-pair encoding implementation with custom tokenizer variants.

#### `minbpe/`
Custom implementation of modern tokenizers inspired by OpenAI's approaches:

**Core Files:**
- `base.py` - Base tokenizer class with common functionality
- `regex.py` - Regex-based tokenizer with configurable splitting patterns
- `gpt4.py` - GPT-4 tokenizer implementation (cl100k_base compatible)

**Key Classes:**
- `Tokenizer` (base.py) - Abstract base class with save/load functionality
- `RegexTokenizer` (regex.py) - BPE tokenizer with regex splitting
- `GPT4Tokenizer` (gpt4.py) - GPT-4 compatible tokenizer

**Features:**
- From-scratch BPE implementation - Complete algorithm implementation
- Regex splitting patterns - GPT-2 and GPT-4 compatible patterns
- Special token support - Handle special tokens like ``
- Save/Load functionality - Persistent model storage
- Unicode handling - Proper UTF-8 byte processing

#### `benchmark.py`
Performance comparison notebook using Marimo:

**Comparisons:**
- Custom minbpe implementation vs tiktoken
- HuggingFace tokenizers vs OpenAI tokenizers
- Encoding/decoding speed benchmarks
- Memory usage analysis
- Tokenization consistency verification

**Metrics:**
- Encoding speed (tokens/second)
- Decoding speed (tokens/second)
- Vocabulary size impact
- Pattern complexity effects

#### `bpe_scratchbook.py`
Interactive exploration notebook for BPE concepts:

**Topics Covered:**
- Byte-level encoding fundamentals
- Unicode and UTF-8 handling
- Pair frequency statistics
- Merge operation visualization
- Token vocabulary building
- Encoding/decoding cycles

**Interactive Features:**
- Step-by-step BPE training visualization
- Token inspection and analysis
- Pattern experimentation
- Performance profiling

### Root Files

#### `notebook.py`
Main exploration notebook for text data concepts:

**Content:**
- Text preprocessing workflows
- Tokenization strategies comparison
- Dataset preparation techniques
- Quality assessment methods

#### `the-verdict.txt`
Sample text dataset for experimentation:
- Classic literature text
- Suitable for tokenizer training
- Benchmarking standard dataset

## Key Concepts

### Byte-Pair Encoding (BPE)
**Algorithm Steps:**
1. Start with byte-level vocabulary (256 tokens)
2. Count frequency of consecutive byte pairs
3. Merge most frequent pair into new token
4. Repeat until desired vocabulary size
5. Use learned merges for encoding/decoding

**Benefits:**
- Handles any Unicode text
- Variable-length tokenization
- Compresses common patterns
- Language-agnostic approach

### Regex Splitting
**Purpose:**
- Intelligently split text before BPE
- Preserve meaningful units (words, numbers)
- Handle punctuation and whitespace
- Improve tokenization quality

**Patterns:**
- GPT-2 pattern: Basic word/number/punctuation splitting
- GPT-4 pattern: Enhanced with contractions and special cases

### Special Tokens
**Common Tokens:**
- `<|endoftext|>` - Document boundary
- `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>` - Fill-in-middle
- `<|start_of_turn|>`, `<|end_of_turn|>` - Conversation turns

**Implementation:**
- Reserved vocabulary indices
- Special handling in encoding/decoding
- Never merged during BPE training

