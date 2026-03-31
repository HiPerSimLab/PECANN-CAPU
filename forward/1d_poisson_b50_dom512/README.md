# PECANN-CAPU: 1D Poisson's Equation with Multi-Scale Solution (b = 50)

## Problem Description

This case applies the **PECANN-CAPU** (Physics-Enforced Constraint Augmented Neural Network — Constrained Augmented Physics-informed Unsupervised) algorithm to a one-dimensional Poisson's equation with a multi-scale solution, originally introduced by Wang et al. [1] and later used by Wang et al. [2] to demonstrate the superior expressive capacity of KANs over standard MLPs.

The governing equation is defined on the domain Ω = {x | 0 ≤ x ≤ 1}:

$$\nabla^2 u = s, \quad \text{in } \Omega$$

$$u = 0, \quad \text{on } \partial\Omega$$

with the exact solution:

$$u(x) = \sin(2\pi x) + 0.1\sin(b\pi x), \quad \forall x \in \Omega$$

where **b = 50** is the user-defined wavenumber. The source function *s(x)* is obtained by substituting the exact solution into the Poisson equation.

## Motivation

This benchmark highlights the severe limitations of conventional PINNs in learning multi-scale solutions. Wang et al. [1] demonstrated that standard PINNs — even those equipped with a single Fourier feature mapping — failed to accurately capture the solution. Only by employing two distinct Fourier feature mappings did they achieve a relative l₂ error norm of 1.36 × 10⁻³.

In contrast, our PECANN-CAPU approach uses only a **single Fourier feature mapping** (replacing the first hidden layer) based on the original method from Tancik et al. [3], and does not require the multiple, tuned Fourier feature mappings adopted in [1].

## Network Architecture & Training

- **Architecture:** 2 hidden layers × 100 units each (wide and shallow, following [1])
- **Fourier feature mapping:** Single mapping matrix **B** ∈ ℝ^(1×50), replacing the first hidden layer
- **Optimizer:** Adam, 40,000 epochs
- **Learning rate schedule:** Exponential decay (factor 0.9 every 1,000 iterations)
- **Collocation points:** 128 domain points, resampled randomly at each epoch

## Directory Structure

```
psn_b50_fourier_dom128/
├── config.py              # Configuration and hyperparameters
├── main.py                # Main training script
├── main.ipynb             # Jupyter notebook version
├── pecann.py              # PECANN module
├── bash.sh                # Shell script
├── run.slurm              # Slurm job submission script
├── core/
│   └── trainer.py         # Training loop and ALM framework
├── data/
│   └── collocation1d.py   # 1D collocation point sampling
├── diagnostics/
│   └── diagnostics.py     # Diagnostic utilities
├── models/
│   └── fc_net.py          # Fully connected network definition
├── physics/
│   └── residuals.py       # PDE and boundary condition residuals
├── utils/
│   ├── io.py              # I/O utilities
│   └── layers.py          # Custom layers (Fourier features, etc.)
├── viz/
│   └── viz.py             # Visualization utilities
├── logs/                  # Training logs and saved models
│   ├── {trial}_*.dat      # Per-trial training metrics
│   └── *.pt               # Saved model checkpoints
└── pic/                   # Output figures
    └── *.png              # Loss curves, solution plots, etc.
```

## References

1. S. Wang, Y. Teng, and P. Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks," *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081, 2021.
2. S. Wang, H. Li, and P. Perdikaris, "KAN or MLP: A fairer comparison," 2024.
3. M. Tancik, P. Srinivasan, B. Mildenhall, et al., "Fourier features let networks learn high frequency functions in low dimensional domains," *NeurIPS*, 2020.
