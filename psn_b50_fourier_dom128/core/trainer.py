import sys
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Callable

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import Config
from pecann import PECANNState, PECANNTrainer

from physics import PDE_opt, boundary_opt
from data import sampling_spatio
from viz import plot_prediction, plot_update
from models import Spatial1DNet, xavier_init
from diagnostics import evaluate_norms_write

from utils import _ensure_dirs, save_spatio_points, append_logs, flush_logs, _torch_device, stats



def make_eval_fn(PDE_opt, b,
                boundary_opt,
                x_dom, x_bc
            ):

    # eval_fn(model) -> objective, constr
    def eval_objective_and_constraints(model, training=True):
        pde_res = PDE_opt(model, x_dom, b, create_graph = training)
        objective = pde_res.square().mean() # avg_pde_loss

        bc_res = boundary_opt(model, x_bc, b) 
        constr = bc_res.square().mean() # avg_bc_loss
        return objective, constr

    return eval_objective_and_constraints


def exponential_decay(iteration):
    decay_rate = 0.9
    decay_step = 1000
    return decay_rate ** (iteration // decay_step)


# -----------------------------
# Trainer
# -----------------------------

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = _torch_device()
        self.eta = torch.tensor(cfg.eta_vec, device=self.device).reshape(-1, 1)
        
        self.b  = torch.tensor(cfg.b, device=self.device)
        
    def build_model(self) -> dict:
        coords_stat = stats(self.cfg.domain_spatio)
        mean_dom_st = torch.from_numpy(coords_stat[0:1, :]).to(self.device)
        stdev_dom_st= torch.from_numpy(coords_stat[1:2, :]).to(self.device)

        layers  = [self.cfg.in_dim] + [self.cfg.w_neurons]*(self.cfg.n_hidden) + [self.cfg.out_dim]

        model = Spatial1DNet(layers, mean_dom_st, stdev_dom_st,
                       fourier_features = self.cfg.fourier_features, sigma = self.cfg.sigma
                      ).to(self.device)
        model.apply(xavier_init)

        return model


    def run(self):
        model = self.build_model()
        print(model)
        print("Total params:", sum(p.numel() for p in model.parameters()))

        l2_trials, linf_trials = [], []
        # Trials
        for trial in range(1, self.cfg.trials+1):
            print(f"current trial: {trial} ")
            model.apply(xavier_init)

            # Optimizer
            optim = torch.optim.Adam(model.parameters())
            using_lbfgs = False
            scheduler = None
            if self.cfg.use_decay_scheduler:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=exponential_decay)

            # Per-trial logs
            first_flush = True   # to control write/append
            mu_buf, lambda_buf, constr_buf, obj_buf = [], [], [], []
            l2_buf, linf_buf = [], []
            
            trainer = PECANNTrainer(
                eval_fn=None,
                eta=self.eta,
            )
            state = PECANNState(
                Lambda=torch.ones((self.cfg.num_lambda, 1), device=self.device),
                Mu=torch.ones((self.cfg.num_lambda, 1), device=self.device),
                Bar_v=torch.zeros((self.cfg.num_lambda, 1), device=self.device),
                previous_loss=torch.tensor(float("inf"), device=self.device),
            )

            # Training loop for this trial
            pbar = tqdm(range(1, self.cfg.epochs + 1), desc=f"Trial {trial}", file=sys.stdout)
            for epoch in pbar:
                # Sampling
                x_dom, x_bc = sampling_spatio(self.cfg)

                # --- create eval function ---
                trainer.eval_fn = eval_fn = make_eval_fn(
                    PDE_opt, self.b,
                    boundary_opt,
                    x_dom, x_bc
                )

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
                                         f"constraints = {constr.item():.3e}")
                    l2, linf = evaluate_norms_write(model, self.cfg, trial, self.device, write=False)
                    l2_buf.append([epoch, l2])
                    linf_buf.append([epoch, linf])
                    
                if epoch % self.cfg.disp2 == 0:
                    print('rela. l2 norm = %.3e, linf norm = %.3e'%(l2, linf))

                    first_flush = flush_logs(trial, mu_buf, lambda_buf, constr_buf, obj_buf, 
                                             l2_buf, linf_buf,
                                             first_flush, self.cfg.out_dir)
                    #plot_update(trial, self.cfg)
                    #l2, linf = evaluate_norms_write(model, self.cfg, trial, self.device, write=True)
                    #plot_prediction(model, self.cfg, trial, self.device)
                    torch.save(model.state_dict(), os.path.join(self.cfg.out_dir, f"{self.cfg.make_method_name()}_{trial}.pt"))

            plot_update(trial, self.cfg)
            l2, linf = evaluate_norms_write(model, self.cfg, trial, self.device, write=True)
            plot_prediction(model, self.cfg, trial, self.device)
            
            l2_trials.append(l2)
            linf_trials.append(linf)

            # Final save
            torch.save(model.state_dict(), os.path.join(self.cfg.out_dir, f"{self.cfg.make_method_name()}_{trial}.pt"))

        print('mean rel L_2: %2.3e' % np.mean(l2_trials))
        print('std  rel L_2: %2.3e' % np.std(l2_trials))
        print('*'*20)
        print('mean L_inf: %2.3e' % np.mean(linf_trials))
        print('std  L_inf: %2.3e' % np.std(linf_trials))
        print('*'*20)
