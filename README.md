# PECANN-CAPU

**Physics and Equality-Constrained Artificial Neural Networks — Conditionally Adaptive Penalty Update**

PECANN-CAPU is a generic Augmented Lagrangian Method (ALM) algorithm for solving partial differential equations (PDEs) with neural networks. It enforces PDE constraints, boundary conditions, and initial conditions through adaptive dual variables (Lagrange multipliers and penalty parameters), eliminating the need for manual loss weight tuning common in conventional Physics-Informed Neural Networks (PINNs).

## Key Features

- **Augmented Lagrangian Method** — automatically balances PDE residuals and boundary/initial condition losses via adaptive Lagrange multipliers (Λ) and penalty parameters (μ)
- **Optimizer agnostic** — supports both Adam and L-BFGS, with optional learning rate scheduling for Adam
- **Modular design** — the core `PECANNTrainer` and `PECANNState` classes are problem-independent; users only need to supply an evaluation function that returns the objective and constraint residuals
- **Lightweight** — minimal dependencies (PyTorch only)

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/QifengHu/PECANN-CAPU.git
```

### From source (editable / development mode)

```bash
git clone https://github.com/QifengHu/PECANN-CAPU.git
cd PECANN-CAPU
pip install -e .
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10

## Quick Start

```python
import torch
from pecann_capu import PECANNTrainer, PECANNState

# 1. Define your model
model = YourNetwork().to(device)

# 2. Define an evaluation function
#    eval_fn(model, training=True) -> (objective, constr)
#      objective: scalar tensor (e.g. PDE residual loss)
#      constr:    1-D tensor of constraint violations (e.g. BC/IC residuals)
def eval_fn(model, training=True):
    objective = compute_pde_residual(model)
    constr = torch.stack([bc_loss_1, bc_loss_2, ...])
    return objective, constr

# 3. Initialize ALM state
num_constraints = 2  # number of BC/IC constraints
state = PECANNState(
    Lambda=torch.zeros(num_constraints, device=device),
    Mu=torch.ones(num_constraints, device=device),
    Bar_v=torch.zeros(num_constraints, device=device),
    previous_loss=torch.tensor(float('inf'), device=device),
)

# 4. Create the trainer
eta = torch.ones(num_constraints, device=device)  # base penalty scale
trainer = PECANNTrainer(eval_fn=eval_fn, eta=eta)

# 5. Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. Training loop
for epoch in range(num_epochs):
    # Primal step
    objective, constr, loss = trainer.step(
        model, optimizer, state, using_lbfgs=False
    )
    # Dual update
    trainer.update_alm(state, constr, loss)
```

## API Reference

### `PECANNState`

Dataclass holding the ALM state that persists across epochs.

| Field | Type | Description |
|-------|------|-------------|
| `Lambda` | `Tensor` | Lagrange multipliers (one per constraint) |
| `Mu` | `Tensor` | Penalty parameters (one per constraint) |
| `Bar_v` | `Tensor` | Exponential moving average of squared constraints |
| `previous_loss` | `Tensor` | Loss from the previous epoch (for dual update condition) |

### `PECANNTrainer`

Main trainer class. Constructed with:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_fn` | — | `eval_fn(model, training) -> (objective, constr)` |
| `eta` | — | Base penalty scale tensor |
| `omega` | 0.999 | Loss improvement threshold for dual update |
| `zeta` | 0.99 | EMA decay rate for constraint variance tracking |
| `eps` | 1e-16 | Small constant for numerical stability |

**Methods:**

- `step(model, optim, state, using_lbfgs, scheduler=None)` — performs one primal optimization step; returns `(objective, constr, loss)`
- `update_alm(state, constr, loss)` — performs the dual variable update

### `PrimalOptimizer`

Static utility class for executing a single optimization step with either Adam or L-BFGS.

## Example Cases

| Case | Description |
|------|-------------|
| `psn_b50_fourier_dom128/` | 1D Poisson equation with multi-scale solution (b=50), Fourier features, 128 domain points |

## Citation

If you use PECANN-CAPU in your research, please cite:

```bibtex
@article{pecann_capu,
  title={PECANN-CAPU: Physics-Enforced Constraint Augmented Neural Network},
  author={Hu, Qifeng},
  year={2025},
  url={https://github.com/QifengHu/PECANN-CAPU}
}
```

## License

MIT
