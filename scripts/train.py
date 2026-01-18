#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main training script for AuON optimizer experiments.

Usage:
    python scripts/train.py --optimizer auon --num-chunks 10 --iterations 5000
    python scripts/train.py --optimizer muon --num-chunks 99 --iterations 10000
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Environment configuration for H100 optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch

# H100 / matmul / cuDNN optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from data.fineweb import download_and_load_data
from training.config import Hyperparameters
from training.trainer import run_training


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT model with AuON or Muon optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auon",
        choices=["auon", "hybrid_auon", "muon"],
        help="Optimizer to use for hidden matrix parameters",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=10,
        help="Number of training data chunks (10 for testing, 99 for full ~10B tokens)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Validate every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for downloaded data (default: ./data/fineweb10B)",
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires GPU.")
        sys.exit(1)
        
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print("✓ H100-optimized setup complete")
    
    # Download data
    print(f"\nDownloading data ({args.num_chunks} chunks)...")
    train_data, val_data = download_and_load_data(
        num_chunks=args.num_chunks,
        data_dir=args.data_dir,
    )
    
    # Configure hyperparameters
    hyperparams = Hyperparameters(
        num_iterations=args.iterations,
        val_loss_every=args.val_every,
    )
    
    # Run training
    print(f"\nStarting training with {args.optimizer.upper()} optimizer...")
    val_steps, val_losses = run_training(
        run_label=args.optimizer.upper(),
        optimizer_type=args.optimizer,
        args=hyperparams,
        train_data=train_data,
        val_data=val_data,
        log_dir=args.log_dir,
        base_seed=args.seed,
    )
    
    # Print final results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {min(val_losses):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
