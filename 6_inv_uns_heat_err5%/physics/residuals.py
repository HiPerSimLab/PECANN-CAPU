#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple, Optional, Callable
from math import pi

import torch

__all__ = [
    "grad_sum",
    "grads_sum",
    "u_exact",
    "PDE_opt",
    "data_opt",
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
    u_e = 5*torch.exp(-3*t)*torch.sin(torch.pi*x)
    return u_e

def k_exact(x,t):
    k_e = 4+x.pow(2)
    return k_e

def q_exact(x, t):
    k   = k_exact(x,t)
    u_x = 5 * torch.pi * torch.exp(-3*t)*torch.cos(torch.pi*x)
    return k*u_x

def h_exact(t):
    return 5 *  torch.exp(-3*t)

def f_exact(x):
    return (-3 + torch.pi**2 *(4+x**2)) * torch.sin(torch.pi*x) - 2*torch.pi * x * torch.cos(torch.pi*x)

def s_exact(x, t):
    return f_exact(x) * h_exact(t)


# -----------------------------
# PDE residual
# -----------------------------
def PDE_opt(model, x: torch.Tensor, t: torch.Tensor,
            create_graph: bool = True
           ):
    u, f    = model(x,t)
    u_x,u_t = grads_sum(u, x, t, create_graph=True)
    
    q     = k_exact(x,t) * u_x
    q_x   = grad_sum(q, x, create_graph=create_graph)
    s     = f * h_exact(t)
    res   = u_t - q_x - s # unsteady heat conduction equation
    return res


# -----------------------------
# Boundary condition
# -----------------------------

def boundary_opt(model, x, t):
    u,_     = model(x,t)
    u_bc    = u_exact(x,t)
    return u - u_bc


# -----------------------------
# Initial condition
# -----------------------------

def initial_opt(model, x, t):
    u,_     = model(x,t)
    u_ic    = u_exact(x,t)
    return u - u_ic


# -----------------------------
# Noisy Data Measurement
# -----------------------------

def data_opt(model,x,t, n=0.):
    u,_     = model(x,t)
    u_da    = u_exact(x,t) * (1 + n*torch.randn_like(u))
    return u - u_da
