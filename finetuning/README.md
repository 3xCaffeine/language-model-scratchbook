# Fine-tuning

This directory contains fine-tuning experiments for adapting pre-trained language models to specific tasks. It covers both classification tasks and instruction following capabilities.

## Overview

Fine-tuning adapts general-purpose language models to specialized tasks through continued training on task-specific datasets. This module demonstrates two major fine-tuning paradigms:

1. **Classification Fine-tuning** - Adapting models for sentiment analysis
2. **Instruction Fine-tuning** - Teaching models to follow specific instructions

## Directory Structure

### `classification/`
Contains implementations for text classification tasks, specifically sentiment analysis on the IMDb dataset.

#### `train_logreg.py`
A baseline implementation using traditional machine learning:
- **Dataset**: IMDb movie reviews (Stanford NLP)
- **Model**: Logistic Regression with bag-of-words features
- **Features**: CountVectorizer for text representation
- **Evaluation**: Accuracy metrics on train/validation/test splits

**Purpose**: Provides a performance baseline for comparison with neural approaches.

#### `train_gpt.py`
Neural approach using fine-tuned GPT models:
- **Dataset**: IMDb movie reviews with preprocessing
- **Model**: GPT architecture adapted for classification
- **Training**: Supervised fine-tuning with cross-entropy loss
- **Features**: Tokenization, padding, and custom dataset handling

**Key Components**:
- `IMDbDataset` class for data loading and preprocessing
- Tokenization using tiktoken
- Sequence padding and truncation
- Training loop with validation

### `instruction/`
Contains instruction fine-tuning for teaching models to follow specific formats and tasks.

#### `instruction_finetuning.py`
Implementation of instruction following capabilities:
- **Dataset**: Alpaca GPT-4 instruction dataset
- **Format**: Structured instruction-input-output triples
- **Training**: Supervised fine-tuning on formatted examples
- **Evaluation**: Generation quality and instruction adherence

**Key Features**:
- `InstructionDataset` class for structured data handling
- Custom formatting for instruction-response pairs
- Tokenization of complete instruction sequences
- Training with teacher forcing

## Key Concepts

### Classification Fine-tuning
- Task Adaptation: Modifying language model for binary/multi-class classification
- Data Preparation: Converting text to tokenized sequences with labels
- Training Strategy: Supervised learning with cross-entropy loss
- Evaluation: Accuracy, precision, recall, F1-score

### Instruction Fine-tuning
- Format Learning: Teaching models specific input/output formats
- Multi-task: Handling various instruction types in single model
- Response Generation: Generating appropriate responses to instructions
- Generalization: Improving model's ability to follow unseen instructions

## Data Handling

### IMDb Dataset
- **Source**: Stanford NLP IMDb dataset
- **Format**: Movie reviews with sentiment labels (positive/negative)
- **Splits**: Train (80%), Validation (20%), Test
- **Preprocessing**: Tokenization, padding, truncation

### Alpaca Dataset
- **Source**: Alpaca GPT-4 instruction dataset
- **Format**: Instruction, input (optional), output triples
- **Structure**: 52K instruction-following examples
- **Formatting**: Special tokens for instruction/response separation

## Training Configuration

### Common Parameters
- **Learning Rate**: Typically 5e-5 to 1e-4 for fine-tuning
- **Batch Size**: 8-32 depending on GPU memory
- **Epochs**: 3-10 for fine-tuning
- **Optimizer**: AdamW with weight decay

### Model Variants
- **GPT-2 Small**: 124M parameters (fast experimentation)
- **GPT-2 Medium**: 355M parameters (better performance)
- **Custom Architectures**: Modified for specific tasks

## Performance Metrics

### Classification
- **Accuracy**: Overall prediction correctness
- **Loss**: Cross-entropy loss
- **Validation**: Performance on held-out data

### Instruction Following
- **Generation Quality**: Coherence and relevance
- **Format Adherence**: Following instruction structure
- **Task Performance**: Success rate on specific tasks

## Dependencies

- PyTorch for neural network training
- Transformers for model architectures
- Tiktoken for tokenization
- Polars for data manipulation
- Scikit-learn for baseline models
- Datasets for data loading