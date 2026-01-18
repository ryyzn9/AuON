# -*- coding: utf-8 -*-
"""
AuON Optimizer: Alternative Unit-norm momentum-updates by Normalized nonlinear scaling.

A linear-time optimizer using cosh-based brake normalization for stable gradient updates.
"""

import torch
from torch import Tensor


class AuON(torch.optim.Optimizer):
    """
    AuON optimizer with cosh-based brake normalization.
    
    This optimizer applies:
    1. Momentum with exponential moving average
    2. Decoupled weight decay
    3. Cosh-based normalization for stable updates
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.24)
        momentum: Momentum factor (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.0)
        gamma: Scaling factor for cosh normalization (default: 1.5)
    
    Example:
        >>> optimizer = AuON(model.parameters(), lr=0.08, momentum=0.95)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.24,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        gamma: float = 1.5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
            
        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            gamma = group["gamma"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                m = state["momentum_buffer"]

                # Decoupled weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p.mul_(1 - eff_weight_decay)

                # Momentum update
                m.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(m, momentum)

                G = grad

                # AuON cosh-based brake normalization
                g_norm = G.norm()
                if g_norm == 0:
                    continue  # Skip if gradient is zero

                g_normalized = G / (g_norm + 1e-7)
                g_scaled = g_normalized * gamma

                # Cosh-based scaling: sqrt(mean(cosh(g_scaled)^2))
                r_scaled = torch.sqrt(torch.mean(torch.cosh(g_scaled) ** 2))
                
                # Final update
                update = G / (r_scaled + 1e-8)
                p.add_(update, alpha=-lr)

        return loss
