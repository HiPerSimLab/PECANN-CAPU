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
    "flux_opt",
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
    mask = x < 0
    u_e[mask] = torch.sin(3 * torch.pi * x[mask]) * t[mask]
    u_e[~mask] = x[~mask] * t[~mask]
    return u_e

def k_exact(x,t):
    k_e = torch.ones_like(x)
    mask = x > 0
    k_e[mask] = 3 * torch.pi
    return k_e

def q_exact(x, t):
    q_e = torch.zeros_like(x)
    mask = x<0
    q_e[mask]  = torch.cos(3 * torch.pi * x[mask]) * 3 * torch.pi * t[mask]
    q_e[~mask] = 3 * torch.pi * t[~mask]
    return q_e

def s_exact(x, t):
    s_e = x.clone().detach()
    mask = x<0
    s_e[mask] = ( 1 + (3*torch.pi)**2 *t[mask] ) * torch.sin( 3*torch.pi*x[mask] )
    return s_e


# -----------------------------
# PDE residual
# -----------------------------
def PDE_opt(model, x: torch.Tensor, t: torch.Tensor,
            create_graph: bool = True
           ):
    u, q  = model(x,t)
    u_t   = grad_sum(u, t, create_graph=True)
    q_x   = grad_sum(q, x, create_graph=create_graph)
    s     = s_exact(x, t)
    res   = u_t - q_x - s # unsteady heat conduction equation
    return res


def flux_opt(model, x: torch.Tensor, t: torch.Tensor,
            create_graph: bool = True
            ):
    u,q   = model(x,t)
    u_x   = grad_sum(u, x, create_graph=create_graph)
    k     = k_exact(x,t)
    res   = q - k*u_x
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
