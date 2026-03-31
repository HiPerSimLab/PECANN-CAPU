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
    "sampling_st",
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

    # left / right (vary t, fix x)
    t_right = draw(n_side) * (ub[1] - lb[1]) + lb[1]
    x_right = torch.full_like(t_right, ub[0])
    t_left  = draw(n_side) * (ub[1] - lb[1]) + lb[1]
    x_left  = torch.full_like(t_left, lb[0])

    x = torch.cat((x_right, x_left), dim=0)
    t = torch.cat((t_right, t_left), dim=0)
    return x, t


def sample_random_initial_points(domain, n_side, dtype, device):
    lb, ub = domain[0], domain[1]
    soboleng = torch.quasirandom.SobolEngine(dimension=1, scramble=True)

    def draw(n):
        return soboleng.draw(n, dtype=dtype).to(device)

    # bottom (vary x, fix t)
    x_bottom = draw(n_side) * (ub[0] - lb[0]) + lb[0]
    t_bottom = torch.full_like(x_bottom, lb[1])

    return x_bottom, t_bottom


# -----------------------------
# Uniform mesh sampling: [129, 129, 11]: interior: 127 points per space side, 10 on time axis (except initial)
# -----------------------------
def sample_uniform_mesh_points(domain, dom_dis, dtype, device, include_side = False):
    lb, ub = domain[0], domain[1]
    x_unique = torch.linspace(lb[0], ub[0], dom_dis[0], dtype=dtype, device=device)
    y_unique = torch.linspace(lb[1], ub[1], dom_dis[1], dtype=dtype, device=device)
    t_unique = torch.linspace(lb[-1], ub[-1], dom_dis[-1], dtype=dtype, device=device)
    if include_side == False:
        tt, yy, xx = torch.meshgrid(t_unique[1:], y_unique[1:-1], x_unique[1:-1], indexing='ij')
    else:
        tt, yy, xx = torch.meshgrid(t_unique, y_unique, x_unique, indexing="ij")
    return xx.reshape(-1, 1), yy.reshape(-1, 1), tt.reshape(-1, 1)


##### boundary: 10 on time axis
def sample_uniform_boundary_points(domain, dom_dis, dtype, device):
    lb, ub = domain[0], domain[1]
    
    x_unique = torch.linspace(lb[0], ub[0], dom_dis[0], dtype=dtype, device=device)
    y_unique = torch.linspace(lb[1], ub[1], dom_dis[1], dtype=dtype, device=device)
    t_unique = torch.linspace(lb[-1], ub[-1], dom_dis[-1], dtype=dtype, device=device)[1:]

    # EAST/WEST boundary: x fixed, vary y and t
    tt_e, yy_e = torch.meshgrid(t_unique, y_unique, indexing='ij')
    xe = torch.full_like(tt_e, ub[0])
    xw = torch.full_like(tt_e, lb[0])
    E_bc = torch.stack((xe, yy_e, tt_e), dim=-1).reshape(-1, 3)
    W_bc = torch.stack((xw, yy_e, tt_e), dim=-1).reshape(-1, 3)

    # NORTH/SOUTH boundary: y fixed, vary x and t
    tt_n, xx_n = torch.meshgrid(t_unique, x_unique[1:-1], indexing='ij')
    yn = torch.full_like(tt_n, ub[1])
    ys = torch.full_like(tt_n, lb[1])
    N_bc = torch.stack((xx_n, yn, tt_n), dim=-1).reshape(-1, 3)
    S_bc = torch.stack((xx_n, ys, tt_n), dim=-1).reshape(-1, 3)

    # Combine all boundary points
    data = torch.cat((E_bc, W_bc, N_bc, S_bc), dim=0)

    return data[:,0:1], data[:,1:2], data[:,2:3]


##### initial: 129*129 spatial points
def sample_uniform_initial_points(domain, dom_dis, dtype, device):
    lb, ub = domain[0], domain[1]
    x_unique = torch.linspace(lb[0], ub[0], dom_dis[0], dtype=dtype, device=device)
    y_unique = torch.linspace(lb[1], ub[1], dom_dis[1], dtype=dtype, device=device)

    # bottom (t=lb)
    xx, yy = torch.meshgrid(x_unique, y_unique, indexing='ij')
    tt = torch.full_like(xx, lb[-1])

    return xx.reshape(-1, 1), yy.reshape(-1, 1), tt.reshape(-1, 1)


# -----------------------------
# Sampling
# -----------------------------
def sampling_st(cfg: Config, domain_win,
                    dtype: torch.dtype = torch.float64,
                    device: torch.device = None):
    if device is None:
        device = _torch_device()

    domain = _to_tensor(domain_win, dtype, device)
        
    if cfg.sampling == "uniform":
        x_dom, y_dom, t_dom = sample_uniform_mesh_points(domain, cfg.dom_dis, dtype, device)
        x_bc, y_bc, t_bc  = sample_uniform_boundary_points(domain, cfg.dom_dis, dtype, device)
        x_ic, y_ic, t_ic  = sample_uniform_initial_points(domain, cfg.dom_dis, dtype, device)
    #elif cfg.sampling == "random":
    #    x_dom, t_dom = sample_random_residual_points(domain, cfg.n_dom, dtype, device)
    #    x_bc,  t_bc  = sample_random_boundary_points(domain, cfg.n_bc, dtype, device)
    #    x_bc,  t_ic  = sample_random_initial_points(domain, cfg.n_ic, dtype, device)
    else:
        raise ValueError(f"Unknown sampling mode: '{cfg.sampling}'")

    x_dom = x_dom.requires_grad_(True); y_dom = y_dom.requires_grad_(True); t_dom = t_dom.requires_grad_(True)
    x_bc  = x_bc.requires_grad_(True);  y_bc  = y_bc.requires_grad_(True);  t_bc  = t_bc.requires_grad_(True)
    x_ic  = x_ic.requires_grad_(True);  y_ic  = y_ic.requires_grad_(True);  t_ic  = t_ic.requires_grad_(True)
    return x_dom, y_dom, t_dom, x_bc, y_bc, t_bc, x_ic, y_ic, t_ic


'''
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
'''




