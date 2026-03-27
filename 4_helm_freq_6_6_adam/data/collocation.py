#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple
import torch
import numpy as np

from config import Config

from utils import _to_tensor, _torch_device

__all__ = [
    "sample_random_residual_points",
    "sample_random_boundary_points",
    "sample_interior_mesh_points",
    "sample_boundary_mesh_points",
    "sampling_spatio",
]


# -----------------------------
# Device / dtype utility
# -----------------------------


# -----------------------------
# Sobol-based random sampling
# -----------------------------
def sample_random_residual_points(domain, n_dom, dtype, device):
    dim = domain.shape[1]
    soboleng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
    data = soboleng.draw(n_dom, dtype=dtype).to(device)
    data = data * (domain[1] - domain[0]) + domain[0]
    x = data[:, 0:1]
    y = data[:, 1:2]
    return x, y


def sample_random_boundary_points(domain, n_side, dtype, device):
    lb, ub = domain[0], domain[1]
    soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble=True)

    def draw(n):
        return soboleng.draw(n, dtype=dtype).to(device)

    # top / bottom (vary x, fix y)
    x_top    = draw(n_side) * (ub[0] - lb[0]) + lb[0]
    y_top    = torch.full_like(x_top, ub[1])
    x_bottom = draw(n_side) * (ub[0] - lb[0]) + lb[0]
    y_bottom = torch.full_like(x_bottom, lb[1])

    # left / right (vary y, fix x)
    y_right = draw(n_side) * (ub[1] - lb[1]) + lb[1]
    x_right = torch.full_like(y_right, ub[0])
    y_left  = draw(n_side) * (ub[1] - lb[1]) + lb[1]
    x_left  = torch.full_like(y_left, lb[0])

    x = torch.cat((x_top, x_right, x_bottom, x_left), dim=0)
    y = torch.cat((y_top, y_right, y_bottom, y_left), dim=0)
    return x, y


# -----------------------------
# Uniform mesh sampling: [51,51] means 51 points per side
# -----------------------------
def sample_uniform_mesh_points(domain, dom_dis, dtype, device, include_bc = False):
    lb, ub = domain[0], domain[1]
    x_unique = torch.linspace(lb[0], ub[0], dom_dis[0], dtype=dtype, device=device)
    y_unique = torch.linspace(lb[1], ub[1], dom_dis[1], dtype=dtype, device=device)
    if include_bc == False:
        yy, xx = torch.meshgrid(y_unique[1:-1], x_unique[1:-1], indexing="ij")
    else:
        yy, xx = torch.meshgrid(y_unique, x_unique, indexing="ij")
    return xx.reshape(-1, 1), yy.reshape(-1, 1)


def sample_boundary_mesh_points(domain, dom_dis, dtype, device):
    lb, ub = domain[0], domain[1]
    x_unique = torch.linspace(lb[0], ub[0], dom_dis[0], dtype=dtype, device=device)
    y_unique = torch.linspace(lb[1], ub[1], dom_dis[1], dtype=dtype, device=device)

    # left (x=lb) and right (x=ub) edges
    y_lr    = y_unique.reshape(-1, 1)
    x_left  = torch.full_like(y_lr, lb[0])
    x_right = torch.full_like(y_lr, ub[0])

    # bottom (y=lb) and top (y=ub) edges — exclude corners to avoid duplicates
    x_tb    = x_unique[1:-1].reshape(-1, 1)
    y_bottom = torch.full_like(x_tb, lb[1])
    y_top    = torch.full_like(x_tb, ub[1])

    x = torch.cat((x_left, x_right, x_tb,     x_tb),   dim=0)
    y = torch.cat((y_lr,   y_lr,    y_bottom,  y_top),  dim=0)
    return x, y


# -----------------------------
# Sampling
# -----------------------------
def sampling_spatio(cfg: Config,
                    dtype: torch.dtype = torch.float64,
                    device: torch.device = None):
    if device is None:
        device = _torch_device()
        
    domain = _to_tensor(cfg.domain_spatio, dtype, device)

    if cfg.sampling == "uniform":
        x_dom, y_dom = sample_uniform_mesh_points(domain, cfg.dom_dis, dtype, device)
        x_bc,  y_bc  = sample_boundary_mesh_points(domain, cfg.dom_dis, dtype, device)
    elif cfg.sampling == "random":
        x_dom, y_dom = sample_random_residual_points(domain, cfg.n_dom, dtype, device)
        x_bc,  y_bc  = sample_random_boundary_points(domain, cfg.n_bc, dtype, device)
    else:
        raise ValueError(f"Unknown sampling mode: '{cfg.sampling}'")

    x_dom = x_dom.requires_grad_(True); y_dom = y_dom.requires_grad_(True)
    x_bc  = x_bc.requires_grad_(True);  y_bc  = y_bc.requires_grad_(True)
    return x_dom, y_dom, x_bc, y_bc





