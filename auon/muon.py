# -*- coding: utf-8 -*-
"""
Muon Optimizer: Momentum with Newton-Schulz orthogonalization.

This optimizer serves as a baseline comparison for AuON.
"""

import torch
from torch import Tensor

from .utils import zeropower_via_newtonschulz5


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization.
    
    Applies momentum followed by Newton-Schulz iterations to produce
    orthogonalized updates.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        weight_decay: Weight decay coefficient (default: 0.01)
        momentum: Momentum factor (default: 0.95)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
    
    Example:
        >>> optimizer = Muon(model.parameters(), lr=0.05, momentum=0.95)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
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
            wd = group["weight_decay"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    
                momentum_buffer = state["momentum_buffer"]
                
                # Effective learning rate with shape scaling
                eff_lr = lr * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                
                # Apply weight decay
                p.mul_(1 - eff_weight_decay)
                
                # Momentum update
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)
                
                # Newton-Schulz orthogonalization
                v = zeropower_via_newtonschulz5(grad.bfloat16(), ns_steps)
                p.add_(other=v, alpha=-eff_lr)

        return loss
