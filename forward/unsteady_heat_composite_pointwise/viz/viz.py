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
    "contour_vel_mag",
    "scatter_st_collocation",
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
_cmap = matplotlib.cm.get_cmap(cmap_list[-1]).reversed()


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
def contour_prediction(model, cfg, trial, device):
    test_dis = cfg.test_dis
    methodname = cfg.make_method_name()
    x_t_u_up = np.loadtxt( os.path.join(cfg.out_dir, f"{methodname}_{trial}_x_t_u_upred_q_qpred.dat") )
    x      = x_t_u_up[:,0:1].reshape(test_dis[1], test_dis[0])
    t      = x_t_u_up[:,1:2].reshape(test_dis[1], test_dis[0])
    u_star = x_t_u_up[:,2:3].reshape(test_dis[1], test_dis[0])
    u_pred = x_t_u_up[:,3:4].reshape(test_dis[1], test_dis[0])
    u_err  = np.abs(u_star - u_pred)
    q_star = x_t_u_up[:,4:5].reshape(test_dis[1], test_dis[0])
    q_pred = x_t_u_up[:,5:6].reshape(test_dis[1], test_dis[0])
    q_err  = np.abs(q_star - q_pred)
    
    varlist = ['u_star', 'u_pred', 'u_err', 'q_star', 'q_pred', 'q_err']
    for k, var in enumerate([u_star, u_pred, u_err, q_star, q_pred, q_err]):
        fig = plt.figure(figsize = (6,5)) #(x,y): 1D: (9,5); 2D: (6,5)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0,0])

        if k == 1:
            vmax = np.max(u_star)
            vmin = np.min(u_star)
        elif k == 4:
            vmax = np.max(q_star)
            vmin = np.min(q_star)
        else:
            vmax = np.max(var)
            vmin = np.min(var)

        pimg=plt.pcolormesh(x, t, var, vmin=vmin, vmax=vmax, cmap=_cmap, shading='gouraud')

        ax.axis('scaled')
        #ax.title.set_text(f'L = {L.cpu().item()}')
        limit = np.ceil(vmax)-1
        _colorbar(pimg,min_val = vmin,max_val= vmax,limit=limit)

        plt.savefig(f'pic/{methodname}_{trial}_{varlist[k]}.png', dpi=300)
        print(f"[viz] Saved pic/{methodname}_{trial}_{varlist[k]}.png")
        plt.close()


def scatter_st_collocation(out_dir, trial, methodname):
    def _load_xy(fname: str):
        path = os.path.join(out_dir, fname)
        arr = np.loadtxt(path) 
        return arr[:, 0:1], arr[:, 1:2]
    
    x_dom, t_dom = _load_xy(f"{trial}_dom.dat")
    x_bc,  t_bc  = _load_xy(f"{trial}_bc.dat")
    x_ic,  t_ic  = _load_xy(f"{trial}_ic.dat")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_dom, t_dom, s=1)
    ax.scatter(x_bc,  t_bc, s=1)
    ax.scatter(x_ic,  t_ic, s=1)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    
    plt.tight_layout()
    plt.savefig(f"pic/{methodname}_{trial}_scatter_st_points.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"[viz] Saved pic/{methodname}_{trial}_scatter_st_points.png")
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
    #num_all = obj_s.shape[0]
    #fig, ax = plt.subplots(figsize=(4, 4))
    #ax.plot(obj_s[:num_all,0], obj_s[:num_all,1], label=r'$\mathcal{J}$', linestyle=linestyles[0], color=colors[0],
    #        linewidth=2, marker=markers[0], markersize=5, markevery=num_all//10)
    #ax.plot(constr_s[:num_all,0], constr_s[:num_all,1], label=r'$\mathcal{C}_{B}$', linestyle=linestyles[1], color=colors[1],
    #        linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    #ax.plot(constr_s[:num_all,0], constr_s[:num_all,2], label=r'$\mathcal{C}_{I}$', linestyle=linestyles[2], color=colors[2],
    #        linewidth=2, marker=markers[2], markersize=5, markevery=num_all//10)
    #ax.plot(constr_s[:num_all,0], constr_s[:num_all,3], label=r'$\mathcal{C}_{Q}$', linestyle=linestyles[3], color=colors[3],
    #        linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    #ax.set_xlabel('Epochs', fontsize=14)
    #ax.set_ylabel("Losses", color="k")
    #ax.semilogy()
    ##ax.set_ylim(1e-10, 1e-2)
    #ax.legend(prop={"size": 12}, frameon=False)
    #plt.tight_layout()
    #plt.grid()
    #plt.savefig(f"pic/{methodname}_{trial}_losses.png", dpi=300)
    #print(f"[viz] Saved pic/{methodname}_{trial}_losses.png")
    #plt.close(fig)

    ##Plot lambda & mu evolution
    #fig, ax = plt.subplots(figsize = (4,4))
    #ax.plot(lambda_s[:num_all,0], lambda_s[:num_all,1], label=r'$\lambda_{B}$', linestyle=linestyles[3], color=colors[3],
    #        linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    #ax.plot(mu_s[:num_all,0], mu_s[:num_all,1], label=r'$\mu_{B}$', linestyle=linestyles[3], color=colors[2],
    #        linewidth=2, marker=markers[3], markersize=5, markevery=num_all//10)
    #
    #ax.plot(lambda_s[:num_all,0], lambda_s[:num_all,2], label=r'$\lambda_{I}$', linestyle=linestyles[1], color=colors[3],
    #        linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    #ax.plot(mu_s[:num_all,0], mu_s[:num_all,2], label=r'$\mu_{I}$', linestyle=linestyles[1], color=colors[2],
    #        linewidth=2, marker=markers[1], markersize=5, markevery=num_all//10)
    #
    #ax.plot(lambda_s[:num_all,0], lambda_s[:num_all,3], label=r'$\lambda_{Q}$', linestyle=linestyles[2], color=colors[3],
    #        linewidth=2, marker=markers[2], markersize=5, markevery=num_all//10)
    #ax.plot(mu_s[:num_all,0], mu_s[:num_all,3], label=r'$\mu_{Q}$', linestyle=linestyles[2], color=colors[2],
    #        linewidth=2, marker=markers[2], markersize=5, markevery=num_all//10)
    #ax.set_xlabel('Epochs', fontsize=14)
    #ax.set_ylabel('Parameters', color='k')
    #ax.semilogy()
    #ax.legend(prop={'size': 12}, frameon=False)
    #plt.tight_layout()
    #plt.grid()
    #plt.savefig(f'pic/{methodname}_{trial}_lambda_mu.png', dpi=300)
    #print(f"[viz] Saved pic/{methodname}_{trial}_lambda_mu.png")
    #plt.close(fig)

    #Plot l2 & linf evolution
    num_all = l2_s.shape[0]
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(l2_s[:num_all,0], l2_s[:num_all,1], label=r'$\mathcal{E}_r(\hat{u}-u)$', linestyle=linestyles[0], color=colors[3],
            linewidth=2)
    ax.plot(l2_s[:num_all,0], l2_s[:num_all,2], label=r'$\mathcal{E}_r(\hat{q}-q)$', linestyle=linestyles[3], color=colors[1],
            linewidth=2)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Norms', color='k')
    ax.semilogy()
    ax.legend(prop={'size': 12}, frameon=False)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'pic/{methodname}_{trial}_l2s.png', dpi=300)
    print(f"[viz] Saved pic/{methodname}_{trial}_l2s.png")
    plt.close(fig)


