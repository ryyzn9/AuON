#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare AuON vs Muon optimizers on the same model and data.

Runs both optimizers sequentially with the same seed for fair comparison,
then generates a comparison plot.

Usage:
    python scripts/compare_optimizers.py --num-chunks 10 --iterations 5000
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import matplotlib.pyplot as plt

# H100 optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from data.fineweb import download_and_load_data
from training.config import Hyperparameters
from training.trainer import run_training


def main():
    parser = argparse.ArgumentParser(
        description="Compare AuON vs Muon optimizers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=10,
        help="Number of training data chunks",
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
        help="Random seed (same for both optimizers)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./val_loss_auon_vs_muon.png",
        help="Output path for comparison plot",
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)
        
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("✓ H100-optimized setup complete")
    
    # Download data
    print(f"\nDownloading data ({args.num_chunks} chunks)...")
    train_data, val_data = download_and_load_data(num_chunks=args.num_chunks)
    
    # Hyperparameters
    hyperparams = Hyperparameters(
        num_iterations=args.iterations,
        val_loss_every=args.val_every,
    )
    
    # Run AuON
    print("\n" + "=" * 50)
    print("Running AuON optimizer...")
    print("=" * 50)
    auon_val_steps, auon_val_losses = run_training(
        run_label="AuON",
        optimizer_type="auon",
        args=hyperparams,
        train_data=train_data,
        val_data=val_data,
        base_seed=args.seed,
    )
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Run Muon
    print("\n" + "=" * 50)
    print("Running Muon optimizer...")
    print("=" * 50)
    muon_val_steps, muon_val_losses = run_training(
        run_label="Muon",
        optimizer_type="muon",
        args=hyperparams,
        train_data=train_data,
        val_data=val_data,
        base_seed=args.seed,  # Same seed for fair comparison
    )
    
    # Plot comparison
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(12, 6))
    
    if auon_val_losses:
        plt.plot(
            auon_val_steps, auon_val_losses,
            marker="o",
            label="AuON val loss",
            linewidth=2,
        )
    if muon_val_losses:
        plt.plot(
            muon_val_steps, muon_val_losses,
            marker="s",
            label="Muon val loss",
            linewidth=2,
        )
    
    plt.xlabel("Step")
    plt.ylabel("Validation loss")
    plt.title("AuON vs Muon: Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved comparison plot to {args.output}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Comparison Summary")
    print("=" * 50)
    print(f"AuON - Final: {auon_val_losses[-1]:.4f}, Best: {min(auon_val_losses):.4f}")
    print(f"Muon - Final: {muon_val_losses[-1]:.4f}, Best: {min(muon_val_losses):.4f}")


if __name__ == "__main__":
    main()
