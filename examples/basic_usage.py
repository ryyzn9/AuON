#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic usage example for AuON optimizer.

This example shows how to use AuON with a simple PyTorch model.
"""

import torch
import torch.nn as nn

# Import AuON optimizer
from auon import AuON, Adam


# Simple example model
class SimpleModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    # Create model
    model = SimpleModel(input_dim=768, hidden_dim=1024, output_dim=10)
    
    # Separate parameters for different optimizers
    # Use Adam for small parameters (biases, layer norms)
    # Use AuON for large matrix parameters (like transformer attention/FFN weights)
    
    small_params = [p for p in model.parameters() if p.dim() < 2]
    matrix_params = [p for p in model.parameters() if p.dim() >= 2]
    
    print(f"Small parameters: {sum(p.numel() for p in small_params):,}")
    print(f"Matrix parameters: {sum(p.numel() for p in matrix_params):,}")
    
    # Create optimizers
    adam_optimizer = Adam(
        small_params,
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    auon_optimizer = AuON(
        matrix_params,
        lr=0.08,
        momentum=0.95,
        weight_decay=0.01,
    )
    
    # Training loop example
    print("\nSimulated training loop:")
    
    for step in range(5):
        # Generate dummy data
        x = torch.randn(32, 768)
        target = torch.randint(0, 10, (32,))
        
        # Forward pass
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        adam_optimizer.step()
        auon_optimizer.step()
        
        # Zero gradients
        adam_optimizer.zero_grad()
        auon_optimizer.zero_grad()
        
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")
    
    print("\n✓ AuON optimizer example complete!")


if __name__ == "__main__":
    main()
