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

def grad_sum(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute d(sum(f))/dx efficiently via grad_outputs of ones."""
    return torch.autograd.grad(
        outputs=f,
        inputs=x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]

def grads_sum(f: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    """Compute (d(sum(f))/dx, d(sum(f))/dy, d(sum(f))/dz)) in one pass each."""
    ones = torch.ones_like(f)
    gx = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    gy = torch.autograd.grad(outputs=f, inputs=y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    gt = torch.autograd.grad(outputs=f, inputs=t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return gx, gy, gt
    

# -----------------------------
# Initial solution
# -----------------------------

def f_initial(x,y,t = 0):
    x0 = 0.5
    y0 = 0.75
    r0 = 0.15

    r = torch.sqrt( (x - x0)**2 + (y - y0)**2 )

    fi = torch.zeros_like(x)
    fi[r<= r0] = 1.
    return fi

# -----------------------------
# PDE residual
# -----------------------------

def PDE_opt(model, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
           T: torch.Tensor, create_graph=True):
    f           = model(x,y,t)
    f_x,f_y,f_t = torch.autograd.grad(f.sum(),(x,y,t),create_graph=create_graph)

    u       = torch.sin(pi*x)**2 * torch.sin(2*pi*y) * torch.cos(pi*t/T)
    v       = - torch.sin(pi*y)**2 * torch.sin(2*pi*x) * torch.cos(pi*t/T)

    res       = f_t + u * f_x + v * f_y - 0 # passive transport equation
    return res


# -----------------------------
# Boundary condition
# -----------------------------

def boundary_opt(model, x,y,t):
    f       = model(x,y,t)
    return f - 0.


# -----------------------------
# Initial condition
# -----------------------------

def initial_opt(model,x,y,t, f_ic=None):
    if f_ic == None:
        f_ic       = f_initial(x,y,t = 0)
    f   = model(x,y,t)
    return f - f_ic