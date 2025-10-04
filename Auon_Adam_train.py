
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import json
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import os
import pickle
from datetime import datetime
import seaborn as sns
from scipy import stats

# --- Imports for Distributed Training ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# ----------------------------------------

warnings.filterwarnings('ignore')

def setup_distributed(rank: int, world_size: int):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # No print here, will be handled by the main process

@dataclass
class ImprovedModelConfig:
    # Model architecture (required fields first)
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    batch_size: int # This will now be PER-GPU batch size
    max_steps: int

    # Training parameters - MUCH MORE AGGRESSIVE
    gradient_accumulation_steps: int = 4  # Simulate larger batches

    # Data parameters - LARGER DATASET
    max_seq_len: int = 512  # Longer sequences
    num_documents: int = 2000  # 5x more documents
    max_tokens: int = 500000  # 8x more tokens

    # Evaluation
    eval_every: int = 500  # Less frequent but more comprehensive
    eval_steps: int = 100  # More validation batches

    # Learning rates (from your search)
    adamw_lr: float = 0.003
    auon_lr: float =  0.24

    # Regularization
    weight_decay: float = 0.1  # Stronger regularization
    dropout: float = 0.1  # Add dropout
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    compile_model: bool = False
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        # 'Small': ImprovedModelConfig(
        #      d_model=384, n_heads=8, n_layers=6, d_ff=1536,
        #      batch_size=24, max_steps=5000
        # ),
        # 'Medium': ImprovedModelConfig(
        #      d_model=512, n_heads=8, n_layers=8, d_ff=2048,
        #      batch_size=16, max_steps=4000
        # ),
        # 'Large': ImprovedModelConfig(
        #      d_model=768, n_heads=16, n_layers=10, d_ff=3072,
        #      batch_size=12, max_steps=3000
        # )
def improved_model_configs():
    """More challenging model configurations"""
    return {
        'Tiny': ImprovedModelConfig(
            d_model=512, n_heads=8, n_layers=6, d_ff=1536,
            batch_size=32, max_steps=6000
        )
    }

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() +  1e-7)
    
    update =X

    x =torch.cosh(update)
    rms = torch.sqrt(torch.mean(x.square()))
    G =((update) *1/ (rms + 1e-8))

    return G
    
class auon(torch.optim.Optimizer):
    """auon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.reshape(p.shape), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

def load_and_cache_data(config: ImprovedModelConfig, rank: int, cache_dir: str = "/kaggle/working/data_cache"):
    """Load and cache tokenized data to avoid reprocessing. Only rank 0 downloads."""

    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()

    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    if rank == 0 and not os.path.exists(cache_file):
        print(f"🔄 [Rank {rank}] Processing new data (will cache for future use)")

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

        texts = []
        for i, item in enumerate(dataset):
            if i >= config.num_documents:
                break
            texts.append(item["text"][:3000])

        print(f"Loaded {len(texts)} documents")
        print("Tokenizing texts...")
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        tokens = all_tokens[:config.max_tokens]
        print(f"Using {len(tokens):,} tokens")
        config.vocab_size = tokenizer.vocab_size

        cached_data = {
            'texts': texts,
            'tokenizer': tokenizer,
            'tokens': tokens
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"💾 [Rank {rank}] Cached data to {cache_file}")
    
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print(f"📦 Loading cached data from {cache_file}")
        
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)

    texts = cached_data['texts']
    tokenizer = cached_data['tokenizer']
    tokens = cached_data['tokens']
    config.vocab_size = tokenizer.vocab_size
    
    if rank == 0:
        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class ImprovedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class ImprovedMinimalLLM(nn.Module):
    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList([
            ImprovedTransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.max_seq_len, config.dropout
            ) for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

class MetricsTracker:
    def __init__(self):
        self.metrics = {}
        self.memory_usage = []

    def log_step(self, step: int, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))

    def log_memory(self, step: int, rank: int):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(rank) / 1e9
            reserved = torch.cuda.memory_reserved(rank) / 1e9
            self.memory_usage.append((step, allocated, reserved))

def setup_optimizer(model: nn.Module, optimizer_type: str, config: ImprovedModelConfig, rank: int):
    """Setup optimizer with optimal learning rates"""

    if optimizer_type == 'auon':
        auon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            clean_name = name.replace('module.', '')
            if (param.ndim == 2 and
                'token_embedding' not in clean_name and
                'norm' not in clean_name and
                param.requires_grad):
                auon_params.append(param)
            else:
                adamw_params.append(param)
        
        if rank == 0:
            print(f"  auon parameters: {sum(p.numel() for p in auon_params):,}")
            print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

        auon_optimizer = auon(auon_params, lr=config.auon_lr, momentum=0.95)
        adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.auon_lr*0.1, weight_decay=config.weight_decay)
        return [auon_optimizer, adamw_optimizer]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.adamw_lr, weight_decay=config.weight_decay)
        return [optimizer]

def comprehensive_evaluate_model(model: nn.Module, val_loader: DataLoader,
                                 config: ImprovedModelConfig, rank: int, world_size: int) -> Dict:
    """Comprehensive evaluation with metrics synchronized across GPUs"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_correct_top5 = 0
    
    device = rank

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

            top5_predictions = logits.topk(5, dim=-1)[1]
            total_correct_top5 += (top5_predictions == y.unsqueeze(-1)).any(dim=-1).sum().item()
    
    if dist.is_initialized():
        metrics = torch.tensor([total_loss, total_tokens, total_correct, total_correct_top5], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_tokens, total_correct, total_correct_top5 = metrics.tolist()

    if rank == 0:
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        top5_accuracy = total_correct_top5 / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(min(avg_loss, 20))

        model.train()
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_top5_accuracy': top5_accuracy,
            'val_perplexity': perplexity
        }
    else:
        model.train()
        return {}


def improved_train_model(optimizer_type: str, config: ImprovedModelConfig,
                         train_loader: DataLoader, val_loader: DataLoader,
                         model_name: str, rank: int, world_size: int, run_id: int = 0) -> Tuple[Optional[MetricsTracker], Optional[Dict]]:
    """Improved training adapted for DDP"""

    if rank == 0:
        print(f"\n🚀 Training {optimizer_type.upper()} on {model_name} (Run {run_id+1})")

    set_seed(42 + run_id * 1000)
    
    model = ImprovedMinimalLLM(config).to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    if config.compile_model:
        if rank == 0: print("  Compiling model...")
        try:
            model_to_compile = model.module if world_size > 1 else model
            compiled_model = torch.compile(model_to_compile, mode='max-autotune')
            if world_size > 1:
                model.module = compiled_model
            else:
                model = compiled_model
            if rank == 0: print("  ✅ Model compiled with max-autotune")
        except Exception as e:
            if rank == 0: print(f"  ⚠️ Compilation failed: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"  📊 Total parameters: {total_params:,}")

    optimizers = setup_optimizer(model, optimizer_type, config, rank)
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None
    tracker = MetricsTracker()

    model.train()
    step = 0
    accumulated_loss = 0
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 2000

    pbar = None
    if rank == 0:
        pbar = tqdm(total=config.max_steps, desc=f"{optimizer_type.upper()}")

    should_break = False
    while step < config.max_steps and not should_break:
        if world_size > 1:
            train_loader.sampler.set_epoch(step)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                should_break = True
                break
            
            x, y = x.to(rank), y.to(rank)
            
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            accumulated_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                
                for optimizer in optimizers:
                    optimizer.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()

                accumulated_loss = 0

            if rank == 0 and step % 50 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))
                tracker.log_step(
                    step,
                    train_loss=current_loss,
                    train_accuracy=accuracy,
                    train_perplexity=perplexity,
                    grad_norm=grad_norm.item() if 'grad_norm' in locals() else 0,
                    learning_rate=optimizers[0].param_groups[0]['lr']
                )
                if step % 500 == 0:
                    tracker.log_memory(step, rank)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            if step > 0 and step % config.eval_every == 0:
                eval_metrics = comprehensive_evaluate_model(model, val_loader, config, rank, world_size)
                
                if world_size > 1:
                    eval_metrics_list = [eval_metrics]
                    dist.broadcast_object_list(eval_metrics_list, src=0)
                    eval_metrics = eval_metrics_list[0]
                
                if rank == 0:
                    for key, value in eval_metrics.items():
                        tracker.log_step(step, **{key: value})

                if eval_metrics and eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                    patience_counter = 0
                else:
                    patience_counter += config.eval_every

                if patience_counter >= patience_limit:
                    if rank == 0:
                        print(f"\n  🛑 Early stopping at step {step} (patience exceeded)")
                    should_break = True
                    break
            
            step += 1
            if rank == 0 and step % 50 == 0:
                pbar.update(50)

    if rank == 0 and pbar:
        pbar.close()
        training_time = time.time() - start_time
        print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    final_eval = comprehensive_evaluate_model(model, val_loader, config, rank, world_size)
    
    if rank == 0:
        print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")
        return tracker, {
            'training_time': training_time,
            'final_metrics': final_eval,
            'best_val_loss': best_val_loss,
            'total_params': total_params,
            'steps_completed': step
        }
    else:
        return None, None

def run_comprehensive_ablation(rank: int, world_size: int, num_runs: int = 3):
    """Run multiple experiments, adapted for DDP."""
    
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    if rank == 0:
        print(f"🚀 COMPREHENSIVE ABLATION: {num_runs} runs per configuration")
        print("="*80)

    model_configs = improved_model_configs()
    all_results = {}

    for model_name, config in model_configs.items():
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"🔬 TESTING {model_name.upper()} MODEL")
            print(f"    Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
            print(f"    Training: {config.max_steps} steps, global batch size {config.batch_size * world_size}")
            print(f"{'='*80}")

        texts, tokenizer, tokens = load_and_cache_data(config, rank)
        dataset = TextTokenDataset(tokens, config.max_seq_len)

        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=2, sampler=train_sampler, pin_memory=True, shuffle=(train_sampler is None))
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=2, sampler=val_sampler, pin_memory=True)
        
        model_results = {'auon': [], 'adamw': []}

        for optimizer_type in ['auon', 'adamw']:
            for run_id in range(num_runs):
                tracker, run_results = improved_train_model(
                    optimizer_type, config, train_loader, val_loader,
                    model_name, rank, world_size, run_id
                )
                
                if world_size > 1:
                    run_data = {'tracker': tracker, **(run_results if run_results else {})}
                    gathered_results = [None] * world_size
                    dist.gather_object(run_data, gathered_results if rank == 0 else None, dst=0)

                    if rank == 0:
                        model_results[optimizer_type].append(gathered_results[0])
                elif rank == 0:
                       model_results[optimizer_type].append({'tracker': tracker, **(run_results if run_results else {})})
        
        if rank == 0:
            all_results[model_name] = {
                'config': config,
                'results': model_results
            }

    if rank == 0:
        save_comprehensive_results(all_results, "/kaggle/working/results", num_runs)
    
    if world_size > 1:
        cleanup_distributed()

# --- NEW PLOTTING AND REPORTING FUNCTIONS ---

def generate_comprehensive_plots(all_results: Dict, output_dir: str):
    """Generates and saves all plots for the experiment."""
    sns.set_theme(style="whitegrid")
    
    for model_name, data in all_results.items():
        config = data['config']
        results = data['results']
        
        # Plotting requires at least one run
        if not results['auon'] or not results['adamw']:
            print(f"Skipping plots for {model_name} due to missing run data.")
            continue
            
        auon_tracker = results['auon'][0]['tracker']
        adamw_tracker = results['adamw'][0]['tracker']

        # 1. Training & Validation Loss
        plt.figure(figsize=(12, 8))
        
        # Unpack train data
        auon_train_steps, auon_train_loss = zip(*auon_tracker.metrics.get('train_loss', []))
        adamw_train_steps, adamw_train_loss = zip(*adamw_tracker.metrics.get('train_loss', []))
        
        # Unpack validation data
        auon_val_steps, auon_val_loss = zip(*auon_tracker.metrics.get('val_loss', []))
        adamw_val_steps, adamw_val_loss = zip(*adamw_tracker.metrics.get('val_loss', []))

        plt.plot(auon_train_steps, pd.Series(auon_train_loss).rolling(10).mean(), label='auon Train Loss (smoothed)', color='blue', alpha=0.6)
        plt.plot(adamw_train_steps, pd.Series(adamw_train_loss).rolling(10).mean(), label='AdamW Train Loss (smoothed)', color='red', alpha=0.6)
        plt.plot(auon_val_steps, auon_val_loss, 'o-', label='auon Validation Loss', color='blue', markersize=5)
        plt.plot(adamw_val_steps, adamw_val_loss, 'o-', label='AdamW Validation Loss', color='red', markersize=5)

        plt.title(f'{model_name}: Training and Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.ylim(bottom=max(0, min(auon_val_loss[-5:] + adamw_val_loss[-5:]) - 0.5))
        plt.savefig(f"{output_dir}/{model_name}_loss_comparison.png", bbox_inches='tight')
        plt.close()

        # 2. Learning Rate and Grad Norm
        fig, ax1 = plt.subplots(figsize=(12, 8))

        auon_lr_steps, auon_lr = zip(*auon_tracker.metrics.get('learning_rate', []))
        auon_gn_steps, auon_gn = zip(*auon_tracker.metrics.get('grad_norm', []))
        adamw_lr_steps, adamw_lr = zip(*adamw_tracker.metrics.get('learning_rate', []))
        adamw_gn_steps, adamw_gn = zip(*adamw_tracker.metrics.get('grad_norm', []))
        
        color = 'tab:blue'
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Learning Rate', color=color)
        ax1.plot(auon_lr_steps, auon_lr, color=color, linestyle='--', label='auon LR')
        ax1.plot(adamw_lr_steps, adamw_lr, color='cyan', linestyle='--', label='AdamW LR')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Gradient Norm', color=color)
        ax2.plot(auon_gn_steps, auon_gn, color=color, alpha=0.7, label='auon Grad Norm')
        ax2.plot(adamw_gn_steps, adamw_gn, color='orange', alpha=0.7, label='AdamW Grad Norm')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.title(f'{model_name}: Learning Rate and Gradient Norm')
        plt.savefig(f"{output_dir}/{model_name}_lr_grad_norm.png", bbox_inches='tight')
        plt.close()

    # 3. Final Metrics Boxplot across all models and runs
    plot_data = []
    for model_name, data in all_results.items():
        for opt in ['auon', 'adamw']:
            for run_result in data['results'][opt]:
                if 'final_metrics' in run_result and run_result['final_metrics']:
                    plot_data.append({
                        'Model': model_name,
                        'Optimizer': opt.upper(),
                        'Final Validation Loss': run_result['final_metrics']['val_loss'],
                        'Final Validation Accuracy': run_result['final_metrics']['val_accuracy']
                    })
    
    if not plot_data:
        print("No final metrics found to generate boxplots.")
        return
        
    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Model', y='Final Validation Loss', hue='Optimizer', data=df)
    plt.title('Final Validation Loss Comparison Across Runs')
    plt.savefig(f"{output_dir}/final_loss_boxplot.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Model', y='Final Validation Accuracy', hue='Optimizer', data=df)
    plt.title('Final Validation Accuracy Comparison Across Runs')
    plt.savefig(f"{output_dir}/final_accuracy_boxplot.png", bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(all_results: Dict, output_dir: str, num_runs: int):
    """Generates a text summary of the experiment results."""
    report_path = f"{output_dir}/comprehensive_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" COMPREHENSIVE ABLATION STUDY REPORT\n")
        f.write("="*80 + "\n\n")

        for model_name, data in all_results.items():
            f.write(f"MODEL: {model_name.upper()}\n")
            f.write("-" * 40 + "\n")
            
            auon_results = data['results']['auon']
            adamw_results = data['results']['adamw']
            
            auon_losses = [r['final_metrics']['val_loss'] for r in auon_results if 'final_metrics' in r and r['final_metrics']]
            adamw_losses = [r['final_metrics']['val_loss'] for r in adamw_results if 'final_metrics' in r and r['final_metrics']]

            auon_accs = [r['final_metrics']['val_accuracy'] for r in auon_results if 'final_metrics' in r and r['final_metrics']]
            adamw_accs = [r['final_metrics']['val_accuracy'] for r in adamw_results if 'final_metrics' in r and r['final_metrics']]
            
            auon_times = [r['training_time'] for r in auon_results if 'training_time' in r]
            adamw_times = [r['training_time'] for r in adamw_results if 'training_time' in r]
            
            f.write(f"Optimizer: auon\n")
            f.write(f"  - Avg Val Loss: {np.mean(auon_losses):.4f} ± {np.std(auon_losses):.4f}\n")
            f.write(f"  - Avg Val Acc:  {np.mean(auon_accs):.4f} ± {np.std(auon_accs):.4f}\n")
            f.write(f"  - Avg Time:     {np.mean(auon_times):.1f}s ± {np.std(auon_times):.1f}s\n\n")
            
            f.write(f"Optimizer: AdamW\n")
            f.write(f"  - Avg Val Loss: {np.mean(adamw_losses):.4f} ± {np.std(adamw_losses):.4f}\n")
            f.write(f"  - Avg Val Acc:  {np.mean(adamw_accs):.4f} ± {np.std(adamw_accs):.4f}\n")
            f.write(f"  - Avg Time:     {np.mean(adamw_times):.1f}s ± {np.std(adamw_times):.1f}s\n\n")

            if num_runs > 1 and len(auon_losses) > 1 and len(adamw_losses) > 1:
                t_stat_loss, p_val_loss = stats.ttest_ind(auon_losses, adamw_losses, equal_var=False)
                t_stat_acc, p_val_acc = stats.ttest_ind(auon_accs, adamw_accs, equal_var=False)
                
                f.write("Statistical Significance (Welch's t-test):\n")
                f.write(f"  - Loss: p-value = {p_val_loss:.4f} ({'Significant' if p_val_loss < 0.05 else 'Not Significant'} at p < 0.05)\n")
                f.write(f"  - Acc:  p-value = {p_val_acc:.4f} ({'Significant' if p_val_acc < 0.05 else 'Not Significant'} at p < 0.05)\n")
            
            f.write("\n" + "="*80 + "\n\n")
    print(f"📊 Comprehensive report saved to {report_path}")
# --- END OF NEW FUNCTIONS ---

def save_comprehensive_results(all_results: Dict, results_dir: str, num_runs: int):
    """Save all results and generate comprehensive analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_results_dir = f"{results_dir}/comprehensive_ablation_{timestamp}"
    os.makedirs(full_results_dir, exist_ok=True)
    print(f"\n💾 Saving comprehensive results to {full_results_dir}")
    
    # Generate plots and a text report from the collected data
    generate_comprehensive_plots(all_results, full_results_dir)
    generate_comprehensive_report(all_results, full_results_dir, num_runs)
    
    with open(f'{full_results_dir}/comprehensive_raw_data.json', 'w') as f:
        # A custom default function to handle non-serializable objects like dataclasses
        def default_serializer(o):
            if isinstance(o, (ImprovedModelConfig, MetricsTracker)):
                return o.__dict__
            return '<not serializable>'
        json.dump(all_results, f, indent=2, default=default_serializer)
    
    print(f"✅ Comprehensive results saved to {full_results_dir}")
    return full_results_dir

def main_worker(rank, world_size, num_runs):
    """The main worker function for each process."""
    run_comprehensive_ablation(rank, world_size, num_runs)

if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    NUM_RUNS = 1

    if WORLD_SIZE > 1:
        print(f"🔍 Found {WORLD_SIZE} GPUs. Spawning processes...")
        mp.spawn(main_worker,
                 args=(WORLD_SIZE, NUM_RUNS),
                 nprocs=WORLD_SIZE,
                 join=True)
    else:
        print("🔍 Found 1 or 0 GPUs. Running in a single process.")
        main_worker(0, 1, NUM_RUNS)
    
    print("\n🎉 ALL TRAINING PROCESSES COMPLETED!")
