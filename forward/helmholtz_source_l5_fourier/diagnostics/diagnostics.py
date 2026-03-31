# diagnostics.py
import torch
import os
import numpy as np
from typing import Tuple

from data import sample_uniform_mesh_points
#from physics import u_exact
from utils import _torch_device, _to_tensor


@torch.no_grad()
def evaluate_norms_write(model, cfg, trial, device,
                         dtype=torch.float64, write=False):
    model.eval()
    test_dis = cfg.test_dis
    
    x_y_u_exact = np.loadtxt(f'./ref_sol/helm_l{int(cfg.L)}_u_ref_nx{test_dis[0]}_ny{test_dis[1]}.dat')
    x_test = torch.from_numpy(x_y_u_exact[:,0:1]).to(device)
    y_test = torch.from_numpy(x_y_u_exact[:,1:2]).to(device)
    u_star = torch.from_numpy(x_y_u_exact[:,2:3]).to(device)
    u_pred = model(x_test, y_test)
    err    = u_star - u_pred

    l2   = torch.norm(err) / torch.norm(u_star)
    linf = torch.max(torch.abs(err))

    if write:
        data = torch.cat((x_test, y_test, u_star, u_pred), dim=1).cpu().numpy()
        filename = os.path.join(cfg.out_dir, f"{cfg.make_method_name()}_{trial}_x_y_u_upred.dat")
        np.savetxt(filename, data, fmt="%.6e")

    return l2.item(), linf.item()


