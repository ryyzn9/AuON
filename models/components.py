# -*- coding: utf-8 -*-
"""
Model components for GPT architecture.

Contains Rotary embeddings, attention layers, MLP, and transformer blocks.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention.flex_attention import BlockMask

from auon.utils import norm


class CastedLinear(nn.Linear):
    """
    Linear layer with optional FP8 precision for training.
    
    Supports both standard linear and FP8 matmul operations for 
    H100-optimized training.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        use_fp8: Whether to use FP8 precision during training
        x_s: Input scale for FP8
        w_s: Weight scale for FP8
        grad_s: Gradient scale for FP8
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_fp8: bool = False,
        x_s: float = 1.0,
        w_s: float = 1.0,
        grad_s: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        """Initialize weights with scaled uniform distribution."""
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(
                _x, self.weight,
                x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s
            )[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    """
    Rotary positional embeddings (RoPE).
    
    Implements rotary position embeddings for transformer attention.
    
    Args:
        dim: Head dimension (must be divisible by 4)
        max_seq_len: Maximum sequence length
    """
    
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j->ij", t, angular_freq)
        
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor) -> Tensor:
        """Apply rotary embeddings to input tensor."""
        assert self.cos.size(0) >= x_BTHD.size(-3)
        
        cos = self.cos[None, :x_BTHD.size(-3), None, :]
        sin = self.sin[None, :x_BTHD.size(-3), None, :]
        
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with rotary embeddings.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        head_dim: Dimension per head (default: 128)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        
        # QKV projection
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()
        
        self.attn_scale = 0.12

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        lambdas: Tensor,
        block_mask: BlockMask,
    ) -> Tensor:
        B, T = x.size(0), x.size(1)
        assert B == 1
        
        # Compute Q, K, V
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(
            B, T, 3 * self.num_heads, self.head_dim
        ).chunk(3, dim=-2)
        
        # Normalize and apply rotary
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        # Value embedding mixing
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v
            
        # Attention via FlashAttention
        q_t = q.transpose(1, 2)  # [B, H, T, D]
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        out = scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )  # [B, H, T, D]

        y = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        
        return y


class MLP(nn.Module):
    """
    Feed-forward network with ReluSquared activation.
    
    Args:
        dim: Input/output dimension
    """
    
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReluSquared activation
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Transformer block with attention and MLP.
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        layer_idx: Layer index (attention is skipped for layer 7)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        layer_idx: int,
    ):
        super().__init__()
        # Attention is skipped for layer 7 (specific architecture design)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(
        self,
        x: Tensor,
        ve: Tensor | None,
        x0: Tensor,
        lambdas: Tensor,
        sa_lambdas: Tensor,
        block_mask: BlockMask,
    ) -> Tensor:
        x = lambdas[0] * x + lambdas[1] * x0
        
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
            
        x = x + self.mlp(norm(x))
        return x
