import sys
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Callable

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import Config
from pecann_capu import PECANNState, PECANNTrainer

from physics import PDE_opt, boundary_opt, initial_opt
from data import sampling_st
from viz import contour_prediction, plot_update, create_video
from models import Sigmoid_FC_Net, xavier_init
from diagnostics import compute_area_conservation_error

from utils import _ensure_dirs, save_st_points, append_logs, flush_logs, _torch_device, stats



def make_eval_fn(PDE_opt, T,
                boundary_opt, initial_opt,
                x_dom, y_dom, t_dom, x_bc, y_bc, t_bc, x_ic, y_ic, t_ic, f_ic
                ):

    # eval_fn(model) -> objective, constr
    def eval_objective_and_constraints(model, training=True):
        pde_res = PDE_opt(model, x_dom, y_dom, t_dom, T, create_graph=training)
        objective = pde_res.square().mean() # avg_pde_loss

        bc_res = boundary_opt(model, x_bc,y_bc,t_bc)
        ic_res = initial_opt(model,  x_ic,y_ic,t_ic, f_ic)
        avg_bc_loss = bc_res.square().mean() 
        avg_ic_loss = ic_res.square().mean() 
        constr = torch.stack([avg_bc_loss, avg_ic_loss]).unsqueeze(-1)
        return objective, constr

    return eval_objective_and_constraints

# -----------------------------
# Trainer
# -----------------------------

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = _torch_device()
        self.eta = torch.tensor(cfg.eta_vec, device=self.device).reshape(-1, 1)
        self.T   = torch.tensor(cfg.T, device=self.device)
        
    def build_model(self) -> dict:
        coords_stat = stats(self.cfg.domain_st)
        mean_dom_st = torch.from_numpy(coords_stat[0:1, :]).to(self.device)
        stdev_dom_st= torch.from_numpy(coords_stat[1:2, :]).to(self.device)

        layers  = [self.cfg.in_dim] + [self.cfg.w_neurons]*(self.cfg.n_hidden) + [self.cfg.out_dim]

        model = Sigmoid_FC_Net(layers, mean_dom_st, stdev_dom_st,
                       fourier_features = self.cfg.fourier_features, sigma = self.cfg.sigma
                      ).to(self.device)
        model.apply(xavier_init)

        return model


    def run(self):
        model = self.build_model()
        print(model)
        print("Total params:", sum(p.numel() for p in model.parameters()))

        # Trials
        for trial in range(1, self.cfg.trials+1):
            f_ic_transfer = None
            domain_win = np.copy(self.cfg.domain_st)
            for i in range(self.cfg.num_win):
                domain_win[0,2] = self.cfg.state_t_array[i]
                domain_win[1,2] = self.cfg.state_t_array[i+1]
                print(f'current time window: from {domain_win[0,-1]:.2f} to {domain_win[1,-1]:.2f}')

                model.apply(xavier_init)

                # Sampling
                x_dom, y_dom, t_dom, x_bc, y_bc, t_bc, x_ic, y_ic, t_ic = sampling_st(self.cfg, domain_win)
                save_st_points(x_dom, y_dom, t_dom, x_bc, y_bc, t_bc, x_ic, y_ic, t_ic, self.cfg.out_dir, trial)

                # Optimizer
                optim = torch.optim.Adam(model.parameters())
                using_lbfgs = False
                scheduler = None
                if self.cfg.use_scheduler:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=100, factor=0.98)
    
                # Per-trial logs
                first_flush = True   # to control write/append
                mu_buf, lambda_buf, constr_buf, obj_buf = [], [], [], []
                
                # --- create eval function ---
                eval_fn = make_eval_fn(
                    PDE_opt, self.T,
                    boundary_opt, initial_opt,
                    x_dom, y_dom, t_dom, x_bc, y_bc, t_bc, x_ic, y_ic, t_ic, f_ic_transfer
                )
    
                trainer = PECANNTrainer(
                    eval_fn=eval_fn,
                    eta=self.eta,
                )
                state = PECANNState(
                    Lambda=torch.ones((self.cfg.num_lambda, 1), device=self.device),
                    Mu=torch.ones((self.cfg.num_lambda, 1), device=self.device),
                    Bar_v=torch.zeros((self.cfg.num_lambda, 1), device=self.device),
                    previous_loss=torch.tensor(float("inf"), device=self.device),
                )

                # Training loop for this trial
                pbar = tqdm(range(1, self.cfg.epochs + 1), desc=f"Win {i+1}", file=sys.stdout)
                for epoch in pbar:
                    if (not using_lbfgs) and self.cfg.switch_to_lbfgs and epoch > self.cfg.lbfgs_start_epoch:
                        optim = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
                        using_lbfgs = True
    
                    objective, constr, loss = trainer.step(
                        model, optim, state, using_lbfgs, scheduler
                    )
    
                    # Logs
                    append_logs(epoch, state, objective, constr,
                                mu_buf, lambda_buf, obj_buf, constr_buf
                    )
                    
                    # CAPU updates
                    trainer.update_alm(state, constr, loss)

                    if self.cfg.print_to_console and (epoch % self.cfg.disp == 0):
                        pbar.set_description(f"epoch {epoch} | "
                                             f"loss {loss.item():.3e} | obj {objective.item():.3e} | "
                                             f"constraints = {', '.join(f'{c.item():.3e}' for c in constr)}")
                        
                    if epoch % self.cfg.disp2 == 0:
                        first_flush = flush_logs(trial, domain_win, mu_buf, lambda_buf, constr_buf, obj_buf, 
                                                 first_flush, self.cfg.out_dir)
    
                        torch.save(model.state_dict(), os.path.join(self.cfg.out_dir, f"{self.cfg.make_method_name()}_{trial}_{domain_win[1,2]:.2f}.pt"))

                t_fc = torch.full_like(x_ic, domain_win[-1,-1])
                f_ic_transfer = model(x_ic, y_ic, t_fc).detach()
                data_fc = torch.cat((x_ic.detach(), y_ic.detach(), t_fc.detach(), f_ic_transfer), dim=1).cpu().numpy()
                filename = os.path.join(self.cfg.out_dir, f"{trial}_final_pred_{domain_win[-1,-1]:.2f}.dat")
                np.savetxt(filename, data_fc, fmt="%.6e")
                error = compute_area_conservation_error(self.cfg, trial, domain_win[-1,-1])
                print(f"Area conservation error: {error:.3e} at {domain_win[-1,-1]:.2f}s")

                plot_update(trial, domain_win, self.cfg)
                contour_prediction(model, domain_win, self.cfg, trial, self.device)
            
                # Final save
                torch.save(model.state_dict(), os.path.join(self.cfg.out_dir, f"{self.cfg.make_method_name()}_{trial}_{domain_win[1,2]:.2f}.pt"))

            create_video(self.cfg, trial, fps=24)

