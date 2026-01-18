# AuON

**A**lternative **U**nit-norm momentum-updates by **N**ormalized nonlinear scaling

A linear-time optimizer that achieves remarkable performance without producing semi-orthogonal matrices while preserving structure to guide better-aligned progress and recondition ill-posed updates.

📄 **Paper**: [A Survey For Linear-time Orthogonal Optimizer](https://ryyzn9.github.io/A-Survey-For-Linear-time-Orthogonal-Optimizer/)

## Features

- **Linear-time complexity**: No expensive SVD or matrix inverse operations
- **Cosh-based normalization**: Stable gradient scaling through hyperbolic cosine
- **Drop-in replacement**: Compatible with PyTorch's optimizer API
- **H100 optimized**: Includes FP8 operations for modern GPU acceleration

## Installation

```bash
# Clone the repository
git clone https://github.com/ryyzn9/AuON.git
cd AuON

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

```python
from auon import AuON, Adam

# Create model
model = YourModel()

# Separate parameters: Adam for small params, AuON for matrix params
small_params = [p for p in model.parameters() if p.dim() < 2]
matrix_params = [p for p in model.parameters() if p.dim() >= 2]

# Create optimizers
adam_optimizer = Adam(small_params, lr=0.008, betas=(0.8, 0.95))
auon_optimizer = AuON(matrix_params, lr=0.08, momentum=0.95, weight_decay=0.01)

# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    
    adam_optimizer.step()
    auon_optimizer.step()
    
    adam_optimizer.zero_grad()
    auon_optimizer.zero_grad()
```

## Training Scripts

### Train with AuON

```bash
python scripts/train.py --optimizer auon --num-chunks 10 --iterations 5000
```

### Compare AuON vs Muon

```bash
python scripts/compare_optimizers.py --num-chunks 10 --iterations 10000
```

## Repository Structure

```
AuON/
├── auon/                    # Core optimizer package
│   ├── optimizer.py         # AuON optimizer
│   ├── muon.py              # Muon optimizer (baseline)
│   ├── adam.py              # Custom Adam
│   └── utils.py             # Newton-Schulz, helpers
│
├── models/                  # Model implementations
│   ├── gpt.py               # GPT model
│   ├── components.py        # Attention, MLP, etc.
│   └── custom_ops.py        # FP8 operations
│
├── data/                    # Data loading
│   └── fineweb.py           # FineWebEDU loader
│
├── training/                # Training infrastructure
│   ├── config.py            # Hyperparameters
│   └── trainer.py           # Training loop
│
├── scripts/                 # Executable scripts
│   ├── train.py             # Main training
│   └── compare_optimizers.py
│
└── examples/                # Usage examples
    └── basic_usage.py
```

## AuON Algorithm

The AuON optimizer applies:

1. **Momentum**: Exponential moving average of gradients
2. **Decoupled weight decay**: Applied before the update step
3. **Cosh normalization**: Scales gradients using `sqrt(mean(cosh(g)²))`

```python
# Core AuON update (simplified)
m = momentum * m + (1 - momentum) * grad
g_normalized = m / (norm(m) + eps)
g_scaled = g_normalized * gamma
r_scaled = sqrt(mean(cosh(g_scaled) ** 2))
update = m / (r_scaled + eps)
param -= lr * update
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.4.0
- CUDA-capable GPU (H100 recommended for FP8 operations)

## Citation

```bibtex
@misc{auon2024,
  title={AuON: Alternative Unit-norm momentum-updates by Normalized nonlinear scaling},
  author={AuON Team},
  year={2024},
  url={https://ryyzn9.github.io/A-Survey-For-Linear-time-Orthogonal-Optimizer/}
}
```

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
