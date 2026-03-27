#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from physics.residuals import exact_sol

__all__ = [
    "plot_prediction",
    "plot_update",
]


# -----------------------------
# Matplotlib config
# -----------------------------

_params = {
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "image.cmap": "gray",
    "axes.grid": False,
    "savefig.dpi": 150,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "text.usetex": False,
    "figure.figsize": [4, 4],
    "font.family": "serif",
}
plt.rcParams.update(_params)

# Default colormap: reversed RdBu (as in your original code)
cmap_list = ['jet','YlGnBu','coolwarm','rainbow','magma','plasma','inferno','Spectral','RdBu']


# -----------------------------
# Small helpers
# -----------------------------


def _colorbar(mappable, min_val, max_val, limit):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    ticks = np.linspace(min_val, max_val, 4, endpoint=True)
    cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cbar.formatter.set_powerlimits((limit, limit))
    plt.sca(last_axes)
    return cbar


def _torch_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_grid(domain: np.array, nx: int, ny: int, device: torch.device, dtype: torch.dtype = torch.float64):
    x = torch.linspace(domain[0][0],domain[1][0], nx, dtype=dtype, device=device)
    y = torch.linspace(domain[0][1],domain[1][1], ny, dtype=dtype, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    return X, Y



# -----------------------------
# Public plotting API
# -----------------------------
def plot_prediction(model, cfg, trial, device):
    test_dis = cfg.test_dis
    methodname = cfg.make_method_name()
    x_u_up = np.loadtxt( os.path.join(cfg.out_dir, f"{methodname}_{trial}_x_u_upred.dat") )
    x      = x_u_up[:,0:1]
    u_star = x_u_up[:,1:2]
    u_pred = x_u_up[:,2:3]
    u_err  = np.abs(u_star - u_pred)
    
    linestyles = ["-", ":", "-.", "--"]
    colors = ["k", "b", "g", "r"]
    markers = ["o", "s", "^", "d"]

    fig, ax = plt.subplots(figsize = (9,5))
    ax.plot(x, u_star, label='Exact', linestyle=linestyles[0], color=colors[3], linewidth=2)
    ax.plot(x, u_pred, label=r'CAPU', linestyle=linestyles[3], color=colors[1], linewidth=2)
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$u$', color='k')
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.savefig(f'pic/{methodname}_{trial}_x_u_upred.png', dpi=300)
    print(f"[viz] Saved pic/{methodname}_{trial}_x_u_upred.png")
    plt.close(fig)
    

def plot_update(trial, cfg):
    """Plot loss/objective, lambda, mu evolutions from 'data' directory."""
    methodname = cfg.make_method_name()
    out_dir = cfg.out_dir    

    obj_s = np.loadtxt(os.path.join(out_dir, f"{trial}_object.dat"))
    mu_s = np.loadtxt(os.path.join(out_dir, f"{trial}_mu.dat"))
    constr_s = np.loadtxt(os.path.join(out_dir, f"{trial}_constr.dat"))
    lambda_s = np.loadtxt(os.path.join(out_dir, f"{trial}_lambda.dat"))
    l2_s   = np.loadtxt(os.path.join(out_dir, f"{trial}_l2.dat"))
    linf_s = np.loadtxt(os.path.join(out_dir, f"{trial}_linf.dat"))
    
    linestyles = ["-", ":", "-.", "--"]
    colors = ["k", "b", "g", "r"]
    markers = ["o", "s", "^", "d"]

    # Loss terms
    num_all = obj_s.shape[0]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(obj_s[:num_all,0], obj_s[:num_all,1], label=r'$\mathcal{J}$', linestyle=linestyles[0], color=colors[0],
            linewidth=2, marker=markers[0], markersize=5, markevery=num_all//10)
    ax.plot(constr_s[:num_all,0], constr_s[:num_all,1], label=r'$\mathcal{C}_{B}$', linestyle=linestyles[1], color=colors[1],
            linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel("Losses", color="k")
    ax.semilogy()
    #ax.set_ylim(1e-10, 1e-2)
    ax.legend(prop={"size": 12}, frameon=False)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"pic/{methodname}_{trial}_losses.png", dpi=300)
    print(f"[viz] Saved pic/{methodname}_{trial}_losses.png")
    plt.close(fig)

    #Plot lambda & mu evolution
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(lambda_s[:num_all,0], lambda_s[:num_all,1], label=r'$\lambda_{B}$', linestyle=linestyles[3], color=colors[3],
            linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    ax.plot(mu_s[:num_all,0], mu_s[:num_all,1], label=r'$\mu_{B}$', linestyle=linestyles[3], color=colors[2],
            linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Parameters', color='k')
    ax.semilogy()
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'pic/{methodname}_{trial}_lambda_mu.png', dpi=300)
    print(f"[viz] Saved pic/{methodname}_{trial}_lambda_mu.png")
    plt.close(fig)

    #Plot l2 & linf evolution
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(l2_s[:num_all,0], l2_s[:num_all,1], label=r'$\mathcal{E}_r$', linestyle=linestyles[0], color=colors[3],
            linewidth=2)
    ax.plot(linf_s[:num_all,0], linf_s[:num_all,1], label=r'$\mathcal{E}_{\infty}$', linestyle=linestyles[3], color=colors[1],
            linewidth=2)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Norms', color='k')
    ax.semilogy()
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'pic/{methodname}_{trial}_l2_linf.png', dpi=300)
    print(f"[viz] Saved pic/{methodname}_{trial}_l2_linf.png")
    plt.close(fig)


