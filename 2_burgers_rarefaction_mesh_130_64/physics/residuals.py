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
    "initial_opt"
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
def u_exact(x,t):
    u_e = torch.zeros_like(x)
    for i in range(x.shape[0]):
        if x[i] <= -0.2*t[i]:
            u_e[i] = -0.2
        elif x[i] < 0.4*t[i]:
            u_e[i] = x[i] / t[i]
        else:
            u_e[i] = 0.4
    return u_e


# -----------------------------
# PDE residual
# -----------------------------
def PDE_opt(model, x: torch.Tensor, t: torch.Tensor,
            create_graph: bool = True
           ):
    u         = model(x,t)
    u_x,u_t   = grads_sum(u, x, t, create_graph=True)
    res       = u_t + u * u_x - 0. # invisid burgers equation
    return res


# -----------------------------
# Boundary condition
# -----------------------------

def boundary_opt(model, x, t):
    u       = model(x,t)
    u_bc    = u_exact(x,t)
    return u - u_bc


# -----------------------------
# Initial condition
# -----------------------------

def initial_opt(model, x, t):
    u       = model(x,t)
    u_ic    = u_exact(x,t)
    return u - u_ic
