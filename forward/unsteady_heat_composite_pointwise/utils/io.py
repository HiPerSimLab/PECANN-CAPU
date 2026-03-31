import torch
import numpy as np

import os

from typing import List, Tuple, Optional, Callable


def _torch_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _to_tensor(domain, dtype, device):
    """Convert numpy domain to torch tensor."""
    if isinstance(domain, np.ndarray):
        return torch.tensor(domain, dtype=dtype, device=device)
    return domain.to(dtype=dtype, device=device)

def _ensure_dirs(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.pic_dir, exist_ok=True)


def save_st_points(x_dom, t_dom, x_bc, t_bc, x_ic, t_ic, out_dir, trial):
    data_dom = torch.cat((x_dom, t_dom), dim=1).detach().cpu().numpy()
    data_bc  = torch.cat((x_bc,  t_bc),  dim=1).detach().cpu().numpy()
    data_ic  = torch.cat((x_ic,  t_ic),  dim=1).detach().cpu().numpy()
    np.savetxt(os.path.join(out_dir, f"{trial}_dom.dat"), data_dom, fmt="%.6e", delimiter=" ")
    np.savetxt(os.path.join(out_dir, f"{trial}_bc.dat"),  data_bc,  fmt="%.6e", delimiter=" ")
    np.savetxt(os.path.join(out_dir, f"{trial}_ic.dat"),  data_ic,  fmt="%.6e", delimiter=" ") 

    
def append_logs(epoch, state, objective, constr,
                mu_buf, lambda_buf, obj_buf, constr_buf):
    e = torch.tensor([epoch], dtype=torch.float64)
    mu_buf.append(torch.cat([e, state.Mu.detach().cpu().view(-1)]))
    lambda_buf.append(torch.cat([e, state.Lambda.detach().cpu().view(-1)]))
    constr_buf.append(torch.cat([e, constr.detach().cpu().view(-1)]))
    obj_buf.append(torch.cat([e, torch.tensor([objective.detach().cpu().item()])]))
    


def flush_logs(trial,
               mu_buf, lambda_buf, constr_buf, obj_buf,
               l2_buf, linf_buf,
               first_flush_flag, out_dir):

    if len(obj_buf) == 0:
        return first_flush_flag  # nothing to write

    mu_arr      = torch.stack(mu_buf).numpy()      # shape: (n_steps, n_mu)
    lambda_arr  = torch.stack(lambda_buf).numpy()  # shape: (n_steps, n_lambda)
    constr_arr  = torch.stack(constr_buf).numpy()  # shape: (n_steps, n_constr)
    obj_arr     = torch.stack(obj_buf).numpy()
    
    l2_arr   = np.array(l2_buf)
    linf_arr = np.array(linf_buf)
    
    mode = "wb" if first_flush_flag else "ab"

    # Use your existing naming convention
    with open(os.path.join(out_dir, f"{trial}_mu.dat"), mode) as f:
        np.savetxt(f, mu_arr, fmt="%.6e", delimiter=" ")

    with open(os.path.join(out_dir, f"{trial}_lambda.dat"), mode) as f:
        np.savetxt(f, lambda_arr, fmt="%.6e", delimiter=" ")

    with open(os.path.join(out_dir, f"{trial}_constr.dat"), mode) as f:
        np.savetxt(f, constr_arr, fmt="%.6e", delimiter=" ")

    with open(os.path.join(out_dir, f"{trial}_object.dat"), mode) as f:
        np.savetxt(f, obj_arr, fmt="%.6e", delimiter=" ")

    with open(os.path.join(out_dir, f"{trial}_l2.dat"), mode) as f:
        np.savetxt(f, l2_arr, fmt="%.6e", delimiter=" ")

    with open(os.path.join(out_dir, f"{trial}_linf.dat"), mode) as f:
        np.savetxt(f, linf_arr, fmt="%.6e", delimiter=" ")
        
    # Clear buffers after writing
    mu_buf.clear()
    lambda_buf.clear()
    constr_buf.clear()
    obj_buf.clear()
    l2_buf.clear()
    linf_buf.clear()
    
    return False  # after first flush, we switch to append mode
