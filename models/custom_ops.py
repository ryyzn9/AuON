# -*- coding: utf-8 -*-
"""
Custom FP8 matmul operations for H100-optimized training.

These operations use torch.library.custom_op to define FP8 matrix multiplication
with custom autograd support.
"""

import torch
from torch import Tensor


def register_fp8_ops():
    """
    Register custom FP8 matmul operations.
    
    Call this function once at the start of training to register the custom ops.
    The operations are registered under the 'nanogpt' namespace.
    """
    pass  # Operations are registered at module import time


@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(
    x: Tensor,
    w: Tensor,
    x_s: float,
    w_s: float,
    grad_s: float
) -> tuple[Tensor, Tensor, Tensor]:
    """
    FP8 matrix multiplication operation.
    
    Performs scaled matrix multiplication using FP8 precision for efficiency.
    
    Args:
        x: Input tensor of shape (N, K)
        w: Weight tensor of shape (M, K)
        x_s: Scale factor for x
        w_s: Scale factor for w
        grad_s: Scale factor for gradient
        
    Returns:
        Tuple of (output, x_fp8, w_fp8)
    """
    assert x.is_contiguous() and w.is_contiguous()
    
    x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
    w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
    
    out = torch._scaled_mm(
        x_f8,
        w_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=x.new_tensor(x_s, dtype=torch.float32),
        scale_b=x.new_tensor(w_s, dtype=torch.float32),
        use_fast_accum=True,
    )
    
    return out, x_f8, w_f8


@mm_op.register_fake
def _mm_op_fake(x: Tensor, w: Tensor, *_):
    """Fake implementation for torch.compile tracing."""
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


def _mm_backward(ctx, grad_out: Tensor, *_):
    """Backward pass for FP8 matmul."""
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    
    grad_inv_s = grad_out.new_tensor(grad_s, dtype=torch.float32)
    x_inv_s = grad_out.new_tensor(x_s, dtype=torch.float32)
    w_inv_s = grad_out.new_tensor(w_s, dtype=torch.float32)
    
    grad_f8 = grad_out.div(grad_s).to(torch.float8_e5m2)
    
    grad_x = torch._scaled_mm(
        grad_f8,
        w_f8.T.contiguous().T,
        out_dtype=torch.bfloat16,
        scale_a=grad_inv_s,
        scale_b=w_inv_s,
        use_fast_accum=False,
    )
    
    grad_w = torch._scaled_mm(
        x_f8.T.contiguous(),
        grad_f8.T.contiguous().T,
        out_dtype=torch.float32,
        scale_a=x_inv_s,
        scale_b=grad_inv_s,
        use_fast_accum=False,
    ).T
    
    return grad_x, grad_w, None, None, None


def _mm_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    """Setup context for backward pass."""
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


# Register autograd
torch.library.register_autograd(mm_op, _mm_backward, setup_context=_mm_setup_context)
