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
def u_exact(x,y, a):
    return torch.sin(a[0] * pi * x) * torch.sin(a[1] * pi * y)

def uxx_exact(x,y, a):
    return - (a[0] * pi).pow(2) * u_exact(x,y, a)

def uyy_exact(x,y, a):
    return - (a[1] * pi).pow(2) * u_exact(x,y, a)

def s_exact(x,y, a, k=1.0):
    return uxx_exact(x,y, a) + uyy_exact(x,y, a) + k**2 * u_exact(x,y, a)


# -----------------------------
# PDE residual
# -----------------------------
def PDE_opt(model, x: torch.Tensor, y: torch.Tensor,
            a: torch.Tensor, k=1.0, create_graph: bool = True
           ):
    u         = model(x,y)
    u_x,u_y   = grads_sum(u, x, y, create_graph=True)
    u_xx      = grad_sum(u_x, x, create_graph=create_graph)
    u_yy      = grad_sum(u_y, y, create_graph=create_graph)
    s         = s_exact(x,y, a, k)
    res       = u_xx + u_yy + k**2 * u - s # helmholtz equation
    return res


# -----------------------------
# Boundary condition
# -----------------------------

def boundary_opt(model, x, y, a):
    u       = model(x,y)
    u_bc    = u_exact(x,y, a)
    return u - u_bc
