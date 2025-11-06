"""
Advanced pretraining script for GPT-2 124M parameter model with:
- HuggingFace smollm-corpus dataset support
- Pre-tokenized dataset caching
- Cosine decay with linear warmup
- Gradient clipping
- Model and optimizer checkpoint saving/loading
- Weights & Biases logging
- Distributed Data Parallel (DDP) support
- Flash Attention via PyTorch's scaled_dot_product_attention
- torch.compile optimization
- GaLore optimizer support
"""

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import time
from typing import Optional, Tuple

import tiktoken
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import wandb
from galore_torch import GaLoreAdamW
from datasets import load_dataset

# Setup project imports
from utils import setup_project_imports

setup_project_imports()

from gpt_model import GPTModel  # noqa: E402

# Import from local gpt_train module
try:
    from gpt_train import calc_loss_batch, evaluate_model, generate_and_print_sample
except ImportError:
    # Try relative import if running as a module
    from pretraining.gpt_train import (
        calc_loss_batch,
        evaluate_model,
        generate_and_print_sample,
    )


# ============================================================================
# Dataset Preparation with HuggingFace smollm-corpus
# ============================================================================


class PreTokenizedDataset(Dataset):
    """Dataset that works with pre-tokenized data."""

    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Use a sliding window to chunk into overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def prepare_and_cache_dataset(
    dataset_name: str,
    dataset_subset: str,
    cache_dir: str,
    tokenizer,
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> str:
    """
    Load HuggingFace dataset, tokenize, and cache the result.
    Returns the path to the cached tokenized data.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'HuggingFaceTB/smollm-corpus')
        dataset_subset: Dataset subset/split (e.g., 'cosmopedia-v2', 'python-edu', 'fineweb-edu-dedup')
        cache_dir: Directory to cache tokenized data
        tokenizer: Tokenizer to use
        max_samples: Maximum number of samples to process (for debugging)
        streaming: Whether to use streaming mode
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create cache filename based on dataset config
    cache_filename = f"tokenized_{dataset_name.replace('/', '_')}_{dataset_subset}.pkl"
    tokenized_cache_file = cache_path / cache_filename
    metadata_file = (
        cache_path / f"metadata_{dataset_name.replace('/', '_')}_{dataset_subset}.json"
    )

    # Check if cache exists
    if tokenized_cache_file.exists() and metadata_file.exists():
        print(f"Loading cached tokenized data from {tokenized_cache_file}")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return str(tokenized_cache_file)

    print(f"Loading dataset: {dataset_name}/{dataset_subset}")

    # Load dataset from HuggingFace
    try:
        dataset = load_dataset(
            dataset_name,
            dataset_subset,
            split="train",
            streaming=streaming,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying without subset...")
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=streaming,
        )

    print("Dataset loaded. Processing and tokenizing...")

    # Process and tokenize
    all_token_ids = []
    total_chars = 0
    num_samples = 0

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Extract text from example (adjust field name based on dataset structure)
        # smollm-corpus typically has 'text' field
        text_data = example.get("text", example.get("content", ""))

        if not text_data:
            continue

        text_data += " <|endoftext|> "  # Add separator
        total_chars += len(text_data)

        # Tokenize
        token_ids = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
        all_token_ids.extend(token_ids)

        num_samples += 1

        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i + 1:,} samples, {len(all_token_ids):,} tokens so far..."
            )

    print("\nTokenization complete!")
    print(f"Total samples: {num_samples:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total tokens: {len(all_token_ids):,}")

    # Save tokenized data
    with open(tokenized_cache_file, "wb") as f:
        pickle.dump(all_token_ids, f)

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "dataset_subset": dataset_subset,
        "num_samples": num_samples,
        "total_tokens": len(all_token_ids),
        "total_chars": total_chars,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Cached tokenized data to {tokenized_cache_file}")

    return str(tokenized_cache_file)


def load_tokenized_dataset(cache_file: str) -> list:
    """Load pre-tokenized dataset from cache."""
    with open(cache_file, "rb") as f:
        token_ids = pickle.load(f)
    return token_ids


def create_dataloaders_from_tokens(
    token_ids,
    train_ratio: float,
    batch_size: int,
    max_length: int,
    stride: int,
    num_workers: int = 0,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """Create train and validation dataloaders from tokenized data."""
    split_idx = int(train_ratio * len(token_ids))

    train_dataset = PreTokenizedDataset(token_ids[:split_idx], max_length, stride)
    val_dataset = PreTokenizedDataset(token_ids[split_idx:], max_length, stride)

    # Create samplers for DDP
    train_sampler = None
    val_sampler = None

    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Shuffle only if not using sampler
        drop_last=True,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


# ============================================================================
# Learning Rate Scheduling
# ============================================================================


def get_lr(
    iteration: int, warmup_iters: int, max_iters: int, max_lr: float, min_lr: float
) -> float:
    """
    Calculate learning rate with linear warmup and cosine decay.

    Args:
        iteration: Current iteration
        warmup_iters: Number of warmup iterations
        max_iters: Maximum number of iterations
        max_lr: Maximum learning rate (after warmup)
        min_lr: Minimum learning rate (at end of decay)
    """
    # Linear warmup
    if iteration < warmup_iters:
        return max_lr * (iteration + 1) / warmup_iters

    # Cosine decay after warmup
    if iteration > max_iters:
        return min_lr

    decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================================
# Checkpoint Management
# ============================================================================


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    tokens_seen: int,
    train_losses: list,
    val_losses: list,
    track_tokens_seen: list,
    output_dir: Path,
    filename: str,
    is_ddp: bool = False,
):
    """Save model and optimizer checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen,
    }

    filepath = output_dir / filename
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")

    return filepath


def load_checkpoint(
    checkpoint_path: str, model, optimizer, device, is_ddp: bool = False
) -> Tuple[int, int, int, list, list, list]:
    """
    Load model and optimizer checkpoint.
    Returns: epoch, global_step, tokens_seen, train_losses, val_losses, track_tokens_seen
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if is_ddp:
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    tokens_seen = checkpoint["tokens_seen"]
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    track_tokens_seen = checkpoint.get("track_tokens_seen", [])

    print(f"Resumed from epoch {epoch}, step {global_step}, tokens {tokens_seen:,}")

    return epoch, global_step, tokens_seen, train_losses, val_losses, track_tokens_seen


# ============================================================================
# DDP Setup
# ============================================================================


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()


# ============================================================================
# Training Loop with All Advanced Features
# ============================================================================


def train_model_advanced(
    model,
    optimizer,
    train_loader,
    val_loader,
    device,
    n_epochs: int,
    eval_freq: int,
    eval_iter: int,
    print_sample_iter: int,
    start_context: str,
    output_dir: Path,
    save_ckpt_freq: int,
    tokenizer,
    max_lr: float = 5e-4,
    min_lr: float = 5e-5,
    warmup_iters: int = 100,
    max_grad_norm: float = 1.0,
    start_epoch: int = 0,
    start_step: int = 0,
    start_tokens: int = 0,
    train_losses: list = None,
    val_losses: list = None,
    track_tokens_seen: list = None,
    use_wandb: bool = False,
    is_ddp: bool = False,
    rank: int = 0,
    train_sampler=None,
):
    """
    Advanced training loop with:
    - Learning rate scheduling (warmup + cosine decay)
    - Gradient clipping
    - Checkpoint saving/loading
    - Weights & Biases logging
    - DDP support
    """
    train_losses = train_losses or []
    val_losses = val_losses or []
    track_tokens_seen = track_tokens_seen or []

    tokens_seen = start_tokens
    global_step = start_step
    start_time = time.time()

    # Calculate max iterations for LR scheduling
    total_iters = n_epochs * len(train_loader)

    # Main process should handle logging
    is_main_process = rank == 0

    try:
        for epoch in range(start_epoch, n_epochs):
            model.train()

            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for input_batch, target_batch in train_loader:
                # Update learning rate with warmup and cosine decay
                lr = get_lr(global_step, warmup_iters, total_iters, max_lr, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Forward pass
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer.step()

                tokens_seen += input_batch.numel()
                global_step += 1

                # Logging and evaluation (main process only)
                if is_main_process:
                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter
                        )
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)

                        elapsed = time.time() - start_time
                        print(
                            f"Ep {epoch + 1} (Step {global_step:06d}): "
                            f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                            f"LR {lr:.2e}, Time {elapsed:.1f}s"
                        )

                        # Log to wandb
                        if use_wandb:
                            wandb.log(
                                {
                                    "epoch": epoch + 1,
                                    "train_loss": train_loss,
                                    "val_loss": val_loss,
                                    "learning_rate": lr,
                                    "tokens_seen": tokens_seen,
                                    "step": global_step,
                                }
                            )

                    # Generate text sample
                    if global_step % print_sample_iter == 0:
                        print("\n" + "=" * 50)
                        print("Sample generation:")
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )
                        print("=" * 50 + "\n")

                    # Save checkpoint
                    if global_step % save_ckpt_freq == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            epoch,
                            global_step,
                            tokens_seen,
                            train_losses,
                            val_losses,
                            track_tokens_seen,
                            output_dir,
                            f"model_step_{global_step}.pt",
                            is_ddp=is_ddp,
                        )

            # End of epoch checkpoint
            if is_main_process:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    tokens_seen,
                    train_losses,
                    val_losses,
                    track_tokens_seen,
                    output_dir,
                    f"model_epoch_{epoch + 1}.pt",
                    is_ddp=is_ddp,
                )

    except KeyboardInterrupt:
        if is_main_process:
            print("\nTraining interrupted by user")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                tokens_seen,
                train_losses,
                val_losses,
                track_tokens_seen,
                output_dir,
                f"model_interrupted_step_{global_step}.pt",
                is_ddp=is_ddp,
            )

    return train_losses, val_losses, track_tokens_seen


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Advanced GPT Model Training with all bells and whistles"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceTB/smollm-corpus",
        help="HuggingFace dataset name to use",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="cosmopedia-v2",
        help="Dataset subset/split (e.g., 'cosmopedia-v2', 'python-edu', 'fineweb-edu-dedup')",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="tokenized_cache",
        help="Directory to cache tokenized data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_checkpoints",
        help="Directory where model checkpoints will be saved",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for dataset loading",
    )

    # Training arguments
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.90,
        help="Ratio of data to use for training vs validation",
    )

    # Optimization arguments
    parser.add_argument(
        "--max_lr", type=float, default=5e-4, help="Maximum learning rate"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=5e-5,
        help="Minimum learning rate (for cosine decay)",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=100, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--use_galore",
        type=bool,
        default=False,
        help="Use GaLore optimizer instead of AdamW",
    )

    # Evaluation and logging arguments
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="Frequency of evaluations during training",
    )
    parser.add_argument(
        "--eval_iter", type=int, default=5, help="Number of iterations for evaluation"
    )
    parser.add_argument(
        "--print_sample_iter",
        type=int,
        default=1000,
        help="Iterations between printing sample outputs",
    )
    parser.add_argument(
        "--save_ckpt_freq",
        type=int,
        default=10000,
        help="Frequency of saving model checkpoints",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Logging arguments
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="gpt-pretraining",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )

    # DDP arguments
    parser.add_argument(
        "--use_ddp", action="store_true", help="Use Distributed Data Parallel training"
    )

    # Optimization arguments
    parser.add_argument(
        "--use_compile", action="store_true", help="Use torch.compile for optimization"
    )

    # Model arguments
    parser.add_argument(
        "--debug", action="store_true", help="Use a very small model for debugging"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )

    args = parser.parse_args()

    # Setup DDP if requested
    rank = 0
    world_size = 1
    is_ddp = args.use_ddp

    if is_ddp:
        rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{rank}")
        print(f"DDP initialized: rank {rank}/{world_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    is_main_process = rank == 0

    # Model configuration
    if args.debug:
        GPT_CONFIG = {
            "vocab_size": 50257,
            "context_length": 256,
            "emb_dim": 128,
            "n_heads": 4,
            "n_layers": 4,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    else:
        GPT_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.1,
            "qkv_bias": False,
        }

    # Initialize wandb (main process only)
    if is_main_process and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **GPT_CONFIG,
                **vars(args),
            },
        )

    # Set random seed
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Prepare or load tokenized dataset (main process does this)
    if is_main_process:
        cache_file = prepare_and_cache_dataset(
            dataset_name=args.dataset_name,
            dataset_subset=args.dataset_subset,
            cache_dir=args.cache_dir,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            streaming=args.streaming,
        )

    # Sync all processes
    if is_ddp:
        dist.barrier()

    # All processes load the tokenized data
    cache_filename = (
        f"tokenized_{args.dataset_name.replace('/', '_')}_{args.dataset_subset}.pkl"
    )
    cache_file = Path(args.cache_dir) / cache_filename
    token_ids = load_tokenized_dataset(str(cache_file))

    if is_main_process:
        print(f"Loaded {len(token_ids):,} tokens from cache")

    # Create dataloaders
    train_loader, val_loader, train_sampler = create_dataloaders_from_tokens(
        token_ids,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        num_workers=args.num_workers,
        use_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    model = GPTModel(GPT_CONFIG)
    model.to(device)

    # Apply optimizations
    if args.use_compile:
        if is_main_process:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap with DDP if requested
    if is_ddp:
        model = DDP(model, device_ids=[rank])

    # Initialize optimizer
    if args.use_galore:
        if is_main_process:
            print("Using GaLore optimizer")
        optimizer = GaLoreAdamW(
            model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
        )

    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    start_tokens = 0
    train_losses = []
    val_losses = []
    track_tokens_seen = []

    if args.resume_from and os.path.exists(args.resume_from):
        (
            start_epoch,
            start_step,
            start_tokens,
            train_losses,
            val_losses,
            track_tokens_seen,
        ) = load_checkpoint(args.resume_from, model, optimizer, device, is_ddp=is_ddp)

    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    if is_main_process:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

    train_losses, val_losses, tokens_seen = train_model_advanced(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        print_sample_iter=args.print_sample_iter,
        start_context="Every effort moves you",
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        tokenizer=tokenizer,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        max_grad_norm=args.max_grad_norm,
        start_epoch=start_epoch,
        start_step=start_step,
        start_tokens=start_tokens,
        train_losses=train_losses,
        val_losses=val_losses,
        track_tokens_seen=track_tokens_seen,
        use_wandb=args.use_wandb,
        is_ddp=is_ddp,
        rank=rank,
        train_sampler=train_sampler,
    )

    # Save final model (main process only)
    if is_main_process:
        final_checkpoint_path = save_checkpoint(
            model,
            optimizer,
            args.n_epochs,
            start_step + len(train_loader) * args.n_epochs,
            tokens_seen[-1] if tokens_seen else 0,
            train_losses,
            val_losses,
            track_tokens_seen,
            output_dir,
            "model_final.pt",
            is_ddp=is_ddp,
        )

        print("\nTraining complete!")
        print(f"Final checkpoint saved to {final_checkpoint_path}")

        if torch.cuda.is_available():
            print(
                f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
            )

        if args.use_wandb:
            wandb.finish()

    # Cleanup DDP
    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
