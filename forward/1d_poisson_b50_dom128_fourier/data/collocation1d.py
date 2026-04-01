#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple
import torch
import numpy as np

from config import Config

from utils import _to_tensor, _torch_device

__all__ = [
    "sample_random_residual_points",
    "sample_uniform_mesh_points",
    "sample_boundary_points",
    "sampling_spatio",
]



# -----------------------------
# Sobol-based random sampling
# -----------------------------
def sample_random_residual_points(domain, n_dom, dtype, device):
    lb, ub = domain[0], domain[1]
    soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble=True)

    def draw(n):
        return soboleng.draw(n, dtype=dtype).to(device)

    # bottom (vary x)
    x_bottom = draw(n_dom) * (ub[0] - lb[0]) + lb[0]

    return x_bottom


# -----------------------------
# Uniform mesh sampling: [128]: interior: 128 points per space side
# -----------------------------
def sample_uniform_mesh_points(domain, dom_dis, dtype, device, include_side = False):
    lb, ub = domain[0], domain[1]
    x_unique = torch.linspace(lb[0], ub[0], dom_dis[0]+2, dtype=dtype, device=device)
    if include_side == False:
        return x_unique[1:-1].reshape(-1, 1)
    else:
        return x_unique.reshape(-1, 1)


def sample_boundary_points(domain, dtype, device):
    lb, ub = domain[0], domain[1]
    x_unique = torch.linspace(lb[0], ub[0], 2, dtype=dtype, device=device)
    return x_unique.reshape(-1, 1)


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
        x_dom = sample_uniform_mesh_points(domain, cfg.dom_dis, dtype, device)
    elif cfg.sampling == "random":
        x_dom = sample_random_residual_points(domain, cfg.n_dom, dtype, device)
    else:
        raise ValueError(f"Unknown sampling mode: '{cfg.sampling}'")

    x_bc  = sample_boundary_points(domain, dtype, device)
    x_dom = x_dom.requires_grad_(True)
    x_bc  = x_bc.requires_grad_(True)
    return x_dom, x_bc




