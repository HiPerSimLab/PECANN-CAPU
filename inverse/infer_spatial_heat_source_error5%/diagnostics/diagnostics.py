# diagnostics.py
import torch
import os
import numpy as np
from typing import Tuple

from data import sample_uniform_mesh_points
from physics import u_exact, f_exact
from utils import _torch_device, _to_tensor


@torch.no_grad()
def evaluate_norms_write(model, cfg, trial, device,
                         dtype=torch.float64, write=False):
    model.eval()
    domain = _to_tensor(cfg.domain_st, dtype, device)

    x_test, t_test = sample_uniform_mesh_points(domain, cfg.test_dis, dtype, device, include_side = True)
    
    u_star = u_exact(x_test, t_test)
    f_star = f_exact(x_test)
    
    u_pred, f_pred = model(x_test, t_test)
    err_u    = u_star - u_pred
    err_f    = f_star - f_pred
    l2_u   = torch.norm(err_u) / torch.norm(u_star)
    l2_f   = torch.norm(err_f) / torch.norm(f_star)
    linf_u = torch.max(torch.abs(err_u))
    linf_f = torch.max(torch.abs(err_f))
    
    if write:
        data = torch.cat((x_test, t_test, u_star, u_pred, f_star, f_pred), dim=1).cpu().numpy()
        filename = os.path.join(cfg.out_dir, f"{cfg.make_method_name()}_{trial}_x_t_u_upred_f_fpred.dat")
        np.savetxt(filename, data, fmt="%.6e")

    return l2_u.item(), l2_f.item(), linf_u.item(), linf_f.item()


