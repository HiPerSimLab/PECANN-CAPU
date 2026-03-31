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

from data import sample_uniform_initial_points

__all__ = [
    "contour_vel_mag",
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

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _colorbar(mappable, min_val, max_val, limit):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    ticks = np.linspace(min_val, max_val, 5, endpoint=True)
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

def contour_prediction(model, domain, cfg, trial, device):
    test_dis = cfg.test_dis
    T = cfg.T
    dtype = torch.float64
    x_tensor, y_tensor, t_tensor = sample_uniform_initial_points(domain, test_dis, dtype, device)
    
    methodname = cfg.make_method_name()
    t_unique = np.linspace(domain[0, -1], domain[1, -1], test_dis[-1])
    for t in t_unique:
        t_tensor = torch.full_like(x_tensor, t)
        f_pred = model(x_tensor, y_tensor, t_tensor).detach()
        
        x_mat = x_tensor.cpu().numpy().reshape(test_dis[0], test_dis[1])
        y_mat = y_tensor.cpu().numpy().reshape(test_dis[0], test_dis[1])
        f_pred_mat = f_pred.cpu().numpy().reshape(test_dis[0], test_dis[1])
        
        # Solve for velocities with arraw
        xa, ya = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        u =  np.sin(np.pi*xa)**2 * np.sin(2*np.pi*ya) * np.cos(np.pi*t/T)
        v = -np.sin(np.pi*ya)**2 * np.sin(2*np.pi*xa) * np.cos(np.pi*t/T)
        magnitudes = np.sqrt(u**2 + v**2)

    #################################### Predicted Solution #########################################
        min_val = 0.
        max_val = 1.
        fig, ax = plt.subplots(1, 1, figsize=(9,9))
        img = plt.pcolormesh(x_mat,y_mat,f_pred_mat, cmap = cmap_list[1],vmin=min_val,vmax=max_val,shading='gouraud')
        scale = 10 * np.sqrt(2) / magnitudes.max()
        plt.quiver(xa, ya, u, v, scale=scale)
        # ax.set_title('$u(x,y)$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        cbar = _colorbar(img,min_val,max_val, 0)
        cbar.formatter.set_powerlimits((0, 0))
        ax.axis('square')
        ax.set_xticks([domain[0][0],(domain[1][0]+ domain[0][0])/2,domain[1][0]])
        ax.set_yticks([domain[0][1],(domain[1][1]+domain[0][1])/2,domain[1][1]])

        plt.figtext(0.500, 0.90,'Time: {:.2f}'.format(t), wrap=True, horizontalalignment='center', fontsize=16)
        plt.savefig(f'pic/{methodname}_{trial}_{t_unique[-1]:.2f}_{t:.2f}s.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.close()



def plot_update(trial, domain_win, cfg):
    """Plot loss/objective, lambda, mu evolutions from 'logs' directory."""
    _ensure_dir("pic")
    methodname = cfg.make_method_name()
    out_dir = cfg.out_dir    
    t1 = domain_win[-1,-1]
    
    obj_s = np.loadtxt(os.path.join(out_dir, f"{trial}_{t1:.2f}_object.dat"))
    mu_s = np.loadtxt(os.path.join(out_dir, f"{trial}_{t1:.2f}_mu.dat"))
    constr_s = np.loadtxt(os.path.join(out_dir, f"{trial}_{t1:.2f}_constr.dat"))
    lambda_s = np.loadtxt(os.path.join(out_dir, f"{trial}_{t1:.2f}_lambda.dat"))
 
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
    ax.plot(constr_s[:num_all,0], constr_s[:num_all,2], label=r'$\mathcal{C}_{I}$', linestyle=linestyles[2], color=colors[2],
            linewidth=2, marker=markers[2], markersize=5, markevery=num_all//10)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel("Losses", color="k")
    ax.semilogy()
    ax.set_ylim(1e-12, 1e0)
    ax.legend(prop={"size": 12}, frameon=False)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"pic/{methodname}_{trial}_{t1:.2f}_losses.png", dpi=300)
    plt.close(fig)

    #Plot lambda & mu evolution
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(lambda_s[:num_all,0], lambda_s[:num_all,1], label=r'$\lambda_{B}$', linestyle=linestyles[3], color=colors[3],
            linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    ax.plot(mu_s[:num_all,0], mu_s[:num_all,1], label=r'$\mu_{B}$', linestyle=linestyles[3], color=colors[2],
            linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    ax.plot(lambda_s[:num_all,0], lambda_s[:num_all,2], label=r'$\lambda_{I}$', linestyle=linestyles[1], color=colors[3],
            linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    ax.plot(mu_s[:num_all,0], mu_s[:num_all,2], label=r'$\mu_{I}$', linestyle=linestyles[1], color=colors[2],
            linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Parameters', color='k')
    ax.semilogy()
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'pic/{methodname}_{trial}_{t1:.2f}_lambda_mu.png', dpi=300)
    plt.close(fig)

