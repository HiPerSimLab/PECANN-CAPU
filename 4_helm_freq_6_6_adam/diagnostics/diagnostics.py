# diagnostics.py
import torch
import os
import numpy as np
from typing import Tuple

from data import sample_uniform_mesh_points
from physics import u_exact
from utils import _torch_device, _to_tensor


@torch.no_grad()
def evaluate_norms_write(model, cfg, trial, device,
                         dtype=torch.float64, write=False):
    model.eval()
    a      = _to_tensor(cfg.a, dtype, device)
    domain = _to_tensor(cfg.domain_spatio, dtype, device)

    x_test, y_test = sample_uniform_mesh_points(domain, cfg.test_dis, dtype, device, include_bc = True)
    
    u_star = u_exact(x_test, y_test, a)
    u_pred = model(x_test, y_test)
    err    = u_star - u_pred

    l2   = torch.norm(err) / torch.norm(u_star)
    linf = torch.max(torch.abs(err))

    if write:
        data = torch.cat((x_test, y_test, u_star, u_pred), dim=1).cpu().numpy()
        filename = os.path.join(cfg.out_dir, f"{cfg.make_method_name()}_{trial}_x_y_u_upred.dat")
        np.savetxt(filename, data, fmt="%.6e")

    return l2.item(), linf.item()


