# -*- coding: utf-8 -*-
"""
Training configuration and hyperparameters.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from auon.utils import next_multiple_of_n


@dataclass
class Hyperparameters:
    """
    Training hyperparameters.
    
    Attributes:
        train_seq_len: Sequence length for training (default: 8192)
        val_seq_len: Sequence length for validation (default: 8192)
        val_tokens: Total validation tokens per evaluation (default: 65536)
        num_iterations: Number of training iterations (default: 10000)
        cooldown_frac: Fraction of training for learning rate cooldown (default: 0.45)
        val_loss_every: Validate every N steps (default: 100)
        save_checkpoint: Whether to save checkpoints (default: False)
        
        # Model architecture
        vocab_size: Vocabulary size (default: 50257)
        num_layers: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 6)
        model_dim: Model dimension (default: 768)
        
        # Optimizer settings (Adam)
        adam_lr: Adam learning rate (default: 0.008)
        adam_betas: Adam beta parameters (default: (0.8, 0.95))
        adam_eps: Adam epsilon (default: 1e-10)
        adam_weight_decay: Adam weight decay (default: 0.0)
        
        # Optimizer settings (AuON/Muon)
        auon_lr: AuON learning rate (default: 0.08)
        auon_momentum: AuON momentum (default: 0.95)
        auon_weight_decay: AuON weight decay (default: 0.01)
        muon_lr: Muon learning rate (default: 0.05)
        muon_momentum: Muon momentum (default: 0.95)
        muon_weight_decay: Muon weight decay (default: 0.01)
    """
    # Training
    train_seq_len: int = 4096 * 2  # 8k tokens
    val_seq_len: int = 4096 * 2
    val_tokens: int = 4096 * 2 * 8  # 8 val chunks
    num_iterations: int = 10000
    cooldown_frac: float = 0.45
    val_loss_every: int = 100
    save_checkpoint: bool = False
    
    # Model
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 6
    model_dim: int = 768
    
    # Adam optimizer
    adam_lr: float = 0.008
    adam_betas: tuple = (0.8, 0.95)
    adam_eps: float = 1e-10
    adam_weight_decay: float = 0.0
    
    # AuON optimizer
    auon_lr: float = 0.08
    auon_momentum: float = 0.95
    auon_weight_decay: float = 0.01
    
    # Muon optimizer
    muon_lr: float = 0.05
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    
    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length for model initialization."""
        return max(self.train_seq_len, self.val_seq_len)


def get_lr(step: int, num_iterations: int, cooldown_frac: float = 0.45) -> float:
    """
    Compute learning rate multiplier with linear cooldown.
    
    Args:
        step: Current training step
        num_iterations: Total number of iterations
        cooldown_frac: Fraction of training for cooldown phase
        
    Returns:
        Learning rate multiplier (1.0 during warmup/plateau, decaying to 0.1)
    """
    x = step / num_iterations
    assert 0 <= x < 1, f"Step fraction must be in [0, 1), got {x}"
    
    if x < 1 - cooldown_frac:
        return 1.0
        
    w = (1 - x) / cooldown_frac
    return w * 1.0 + (1 - w) * 0.1


def get_window_size_blocks(
    step: int,
    num_iterations: int,
    device: str = "cuda",
) -> Tensor:
    """
    Compute sliding window size in blocks for attention.
    
    The window size grows linearly from 512 to ~1024 during training.
    
    Args:
        step: Current training step
        num_iterations: Total number of iterations
        device: Device for the output tensor
        
    Returns:
        Number of blocks as int32 tensor
    """
    x = step / num_iterations
    assert 0 <= x <= 1, f"Step fraction must be in [0, 1], got {x}"
    
    window_size = next_multiple_of_n(512 * x + 512, n=128)
    return torch.tensor(window_size // 128, dtype=torch.int32, device=device)
