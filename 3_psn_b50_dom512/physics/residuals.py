#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple, Optional, Callable
from math import pi

import torch

__all__ = [
    "grad_sum",
    "grads_sum",
    "exact_sol",
    "PDE_opt",
    "boundary_opt",
]


# -----------------------------
# Gradient utilities
# -----------------------------

def grad_sum(f: torch.Tensor, x: torch.Tensor,
            create_graph: bool = True) -> torch.Tensor:
    """Compute d(sum(f))/dx efficiently via grad_outputs of ones."""
    return torch.autograd.grad(
        outputs=f.sum(),
        inputs=x,
        create_graph=create_graph,
        retain_graph=True
    )[0]

def grads_sum(f: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
            create_graph: bool = True):
    """Compute (d(sum(f))/dx, d(sum(f))/dy) in one pass each."""
    gx, gy = torch.autograd.grad(f.sum(), (x,y), create_graph=create_graph, retain_graph=True)
    return gx, gy

    
    
# -----------------------------
# Exact solution
# -----------------------------

def u_exact(x, b):
    u_e = torch.sin(2 * pi * x) + 0.1 * torch.sin(b * pi * x)
    return u_e

def uxx_exact(x, b):
    term1 = - (2 * pi)**2 * torch.sin(2 * pi * x)
    term2 = - 0.1 * (b * pi)**2 * torch.sin(b * pi * x)
    return term1 + term2


# -----------------------------
# PDE residual
# -----------------------------
def PDE_opt(model, x: torch.Tensor, 
           b, create_graph: bool = True ):
    u       = model(x)
    u_x     = grad_sum(u, x)
    u_xx    = grad_sum(u_x, x, create_graph = create_graph)
    res     = u_xx - uxx_exact(x, b) # poisson's equation
    return res


# -----------------------------
# Boundary condition
# -----------------------------

def boundary_opt(model, x, b):
    u       = model(x)
    u_bc    = u_exact(x, b)
    return u - u_bc
