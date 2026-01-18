# -*- coding: utf-8 -*-
"""
Training utilities and main training loop.
"""

import time
import uuid
from pathlib import Path
from typing import Type, Tuple, List

import torch
from torch import nn, Tensor
from tqdm.auto import tqdm

from auon import AuON, HybridAuON, Muon, Adam
from models.gpt import GPT
from data.fineweb import data_generator
from .config import Hyperparameters, get_lr, get_window_size_blocks


def setup_model(args: Hyperparameters, device: str = "cuda") -> nn.Module:
    """
    Initialize and configure the GPT model.
    
    Args:
        args: Hyperparameters configuration
        device: Device to place model on
        
    Returns:
        Configured GPT model
    """
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        model_dim=args.model_dim,
        max_seq_len=args.max_seq_len,
    ).to(device)

    # Convert embeddings to bfloat16 for efficiency
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

    return model


def setup_optimizers(
    model: nn.Module,
    args: Hyperparameters,
    optimizer_type: str = "auon",
) -> List[torch.optim.Optimizer]:
    """
    Set up optimizers for training.
    
    Uses Adam for scalar/embedding/head parameters and the specified
    optimizer (AuON, HybridAuON, or Muon) for hidden matrix parameters.
    
    Args:
        model: GPT model
        args: Hyperparameters configuration
        optimizer_type: "auon", "hybrid_auon", or "muon"
        
    Returns:
        List of optimizers [adam_optimizer, main_optimizer]
    """
    # Separate parameter groups
    hidden_matrix_params = [
        p for n, p in model.blocks.named_parameters()
        if p.ndim >= 2 and "embed" not in n
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # Adam for scalars, embeddings, and head
    optimizer1 = Adam(
        scalar_params + head_params + embed_params,
        lr=args.adam_lr,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
    )

    # Main optimizer for hidden matrices
    if optimizer_type.lower() == "auon":
        optimizer2 = AuON(
            hidden_matrix_params,
            lr=args.auon_lr,
            momentum=args.auon_momentum,
            weight_decay=args.auon_weight_decay,
        )
    elif optimizer_type.lower() in ("hybrid_auon", "hybrid"):
        optimizer2 = HybridAuON(
            hidden_matrix_params,
            lr=args.auon_lr,
            momentum=args.auon_momentum,
            weight_decay=args.auon_weight_decay,
            ns_steps=5,
        )
    elif optimizer_type.lower() == "muon":
        optimizer2 = Muon(
            hidden_matrix_params,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.muon_weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. "
                         f"Valid options: 'auon', 'hybrid_auon', 'muon'")

    optimizers = [optimizer1, optimizer2]

    # Store initial learning rates
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

    return optimizers


def run_training(
    run_label: str,
    optimizer_type: str,
    args: Hyperparameters,
    train_data: Tensor,
    val_data: Tensor,
    log_dir: Path | str = "./logs",
    base_seed: int = 1234,
) -> Tuple[List[int], List[float]]:
    """
    Run a complete training loop.
    
    Args:
        run_label: Label for this run (e.g., "AuON" or "Muon")
        optimizer_type: "auon" or "muon"
        args: Hyperparameters configuration
        train_data: Training data tensor
        val_data: Validation data tensor
        log_dir: Directory for logs
        base_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (val_steps, val_losses) for plotting
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    # Setup model and optimizers
    model = setup_model(args)
    optimizers = setup_optimizers(model, args, optimizer_type)

    # Logging
    run_id = uuid.uuid4()
    logfile = log_dir / f"{run_label}_{run_id}.txt"

    def print0(s: str, console: bool = False):
        with open(logfile, "a") as f:
            if console:
                try:
                    tqdm.write(s)
                except Exception:
                    print(s)
            print(s, file=f)

    print0("=" * 100)
    print0(f"run_label: {run_label}")
    print0(f"run_id: {run_id}")
    print0(f"train tokens: {len(train_data):,}")
    print0(f"val tokens: {len(val_data):,}")
    try:
        print0(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        print0("GPU: unknown")
    print0(f"train_seq_len: {args.train_seq_len:,} val_seq_len: {args.val_seq_len:,}")
    print0(f"Optimizer1 (Adam): lr={args.adam_lr}")
    print0(f"Optimizer2 ({run_label}): type={optimizer_type}")
    print0("=" * 100, console=True)

    # Data loaders
    train_loader = data_generator(train_data, args.train_seq_len, align_to_bos=True)
    val_loader = data_generator(val_data, args.val_seq_len, align_to_bos=False)

    # Warmup
    warmup_steps = 1
    print0(f"[{run_label}] Warmup {warmup_steps} steps...")
    for _ in range(warmup_steps):
        inputs, targets = next(train_loader)
        loss = model(inputs, targets, get_window_size_blocks(0, args.num_iterations))
        loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    print0(f"[{run_label}] Warmup done.")

    # Training loop
    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_steps = args.num_iterations

    print0(f"[{run_label}] Starting training...")

    last_val_loss = None
    val_steps_list, val_losses_list = [], []

    with tqdm(range(train_steps + 1), desc=f"Training ({run_label})", total=train_steps + 1) as pbar:
        for step in pbar:
            last_step = (step == train_steps)

            # Validation
            if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.perf_counter() - t0)
                model.eval()

                val_batch_size = args.val_seq_len
                vsteps = min(
                    args.val_tokens // val_batch_size,
                    len(val_data) // val_batch_size,
                )
                val_loss_accum = 0.0

                with torch.no_grad():
                    with tqdm(
                        range(vsteps),
                        desc=f"Validating {run_label} (step {step})",
                        leave=False,
                        total=vsteps,
                    ) as vbar:
                        for i in vbar:
                            inputs, targets = next(val_loader)
                            bl = model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
                            bl_val = float(bl.item())
                            val_loss_accum += bl_val
                            running = val_loss_accum / (i + 1)
                            vbar.set_postfix({"loss": f"{running:.4f}"})

                val_loss = val_loss_accum / max(vsteps, 1)
                last_val_loss = float(val_loss)
                val_steps_list.append(step)
                val_losses_list.append(last_val_loss)

                msg = (
                    f"[{run_label}] step:{step}/{train_steps} "
                    f"val_loss:{val_loss:.4f} "
                    f"train_time:{training_time_ms:.0f}ms "
                    f"step_avg:{training_time_ms/max(step,1):.2f}ms"
                )
                print0(msg, console=True)

                pbar.set_postfix({
                    "val_loss": f"{last_val_loss:.4f}",
                    "train_time_ms": f"{training_time_ms:.0f}",
                })

                model.train()
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            if last_step:
                break

            # Training step
            inputs, targets = next(train_loader)
            loss = model(inputs, targets, get_window_size_blocks(step, args.num_iterations))
            loss.backward()

            # Update learning rates
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * get_lr(step, args.num_iterations, args.cooldown_frac)

            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)

            # Update progress bar
            approx_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            post = {
                "train_time_ms": f"{approx_ms:.0f}",
                "step_avg_ms": f"{approx_ms/(step+1):.2f}",
            }
            if last_val_loss is not None:
                post["val_loss"] = f"{last_val_loss:.4f}"
            pbar.set_postfix(post)

    torch.cuda.synchronize()
    print0(
        f"[{run_label}] peak mem: "
        f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB",
        console=True,
    )
    print0(f"[{run_label}] Training complete.", console=True)

    return val_steps_list, val_losses_list
