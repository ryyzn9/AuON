# -*- coding: utf-8 -*-
"""
Utility functions for AuON optimizers.

Contains Newton-Schulz iteration and other helper functions.
"""

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Compute the zeroth power / orthogonalization of G via Newton-Schulz iteration.
    
    This function applies Newton-Schulz iterations to compute an orthogonal
    approximation of the input tensor G. Used by the Muon optimizer.
    
    Args:
        G: Input tensor of shape (..., M, N), must have ndim >= 2
        steps: Number of Newton-Schulz iterations (default: 5)
        
    Returns:
        Orthogonalized tensor of the same shape as G
        
    Note:
        The iteration is performed on the "wider" dimension to ensure
        numerical stability: if M > N, we transpose before iterating.
    """
    assert G.ndim >= 2, "Input tensor must have at least 2 dimensions"
    
    # Newton-Schulz coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G
    
    # Work with tall shape: (#rows <= #cols)
    if G.size(-2) > G.size(-1):
        X = X.mT
        
    # Normalize by Frobenius norm
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    # Transpose back if needed
    if G.size(-2) > G.size(-1):
        X = X.mT
        
    return X


def next_multiple_of_n(v: float | int, *, n: int) -> int:
    """
    Find the next multiple of n that is >= v.
    
    Args:
        v: Value to round up
        n: The multiple to align to
        
    Returns:
        The smallest multiple of n that is >= v
        
    Example:
        >>> next_multiple_of_n(100, n=128)
        128
        >>> next_multiple_of_n(256, n=128)
        256
    """
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def norm(x: Tensor) -> Tensor:
    """
    Apply RMS normalization to the input tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        RMS-normalized tensor
    """
    import torch.nn.functional as F
    return F.rms_norm(x, (x.size(-1),))
