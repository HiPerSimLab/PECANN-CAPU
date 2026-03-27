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

from physics import PDE_opt, boundary_opt
from data import sampling_spatio
from viz import contour_prediction, plot_update, scatter_spatio_collocation
from models import SpatialNet, xavier_init
from diagnostics import evaluate_norms_write

from utils import _ensure_dirs, save_spatio_points, append_logs, flush_logs, _torch_device, stats



def make_eval_fn(PDE_opt, a,
                 boundary_opt,
                 x_dom,y_dom, x_bc,y_bc):

    # eval_fn(model) -> objective, constr
    def eval_objective_and_constraints(model, training=True):
        pde_res = PDE_opt(model, x_dom, y_dom, a, create_graph=training)
        objective = pde_res.square().mean() # avg_pde_loss

        bc_res = boundary_opt(model, x_bc,y_bc, a)
        constr = bc_res.square().mean() # avg_bc_loss
        
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
        self.a = torch.tensor(self.cfg.a, device=self.device)
        
    def build_model(self) -> dict:
        coords_stat = stats(self.cfg.domain_spatio)
        mean_dom_spatio = torch.from_numpy(coords_stat[0:1, :]).to(self.device)
        stdev_dom_spatio= torch.from_numpy(coords_stat[1:2, :]).to(self.device)

        layers  = [self.cfg.in_dim] + [self.cfg.w_neurons]*(self.cfg.n_hidden) + [self.cfg.out_dim]

        model = SpatialNet(layers, mean_dom_spatio, stdev_dom_spatio,
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

            # Sampling
            x_dom,y_dom, x_bc,y_bc = sampling_spatio(self.cfg)
            save_spatio_points(x_dom,y_dom, x_bc,y_bc, self.cfg.out_dir, trial)
            scatter_spatio_collocation(self.cfg.out_dir, trial, self.cfg.make_method_name())

            # Optimizer
            optim = torch.optim.Adam(model.parameters())
            using_lbfgs = False
            scheduler = None
            if self.cfg.use_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=100, factor=0.98)

            # Per-trial logs
            first_flush = True   # to control write/append
            mu_buf, lambda_buf, constr_buf, obj_buf = [], [], [], []
            l2_buf, linf_buf = [], []
            
            # --- create eval function ---
            eval_fn = make_eval_fn(
                PDE_opt, self.a,
                boundary_opt,
                x_dom,y_dom, x_bc,y_bc
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
            pbar = tqdm(range(1, self.cfg.epochs + 1), desc=f"Trial {trial}", file=sys.stdout)
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
                                         f"constraints = {constr.item():.3e}")
                    l2, linf = evaluate_norms_write(model, self.cfg, trial, self.device, write=False)
                    l2_buf.append([epoch, l2])
                    linf_buf.append([epoch, linf])
                    
                if epoch % self.cfg.disp2 == 0:
                    #l2, linf = evaluate_norms_write(model, self.cfg, trial, self.device, write=True)
                    print('rela. l2 norm = %.3e, linf norm = %.3e'%(l2, linf))

                    first_flush = flush_logs(trial, mu_buf, lambda_buf, constr_buf, obj_buf, 
                                             l2_buf, linf_buf,
                                             first_flush, self.cfg.out_dir)

                    #plot_update(trial, self.cfg)
                    #contour_prediction(model, self.cfg, trial, self.device)
                    torch.save(model.state_dict(), os.path.join(self.cfg.out_dir, f"{self.cfg.make_method_name()}_{trial}.pt"))

            plot_update(trial, self.cfg)
            l2, linf = evaluate_norms_write(model, self.cfg, trial, self.device, write=True)
            contour_prediction(model, self.cfg, trial, self.device)
            
            l2_trials.append(l2)
            linf_trials.append(linf)

            # Final save
            torch.save(model.state_dict(), os.path.join(self.cfg.out_dir, f"{self.cfg.make_method_name()}_{trial}.pt"))

        #print(l2_trials)
        print('mean rel L_2: %2.3e' % np.mean(l2_trials))
        print('std  rel L_2: %2.3e' % np.std(l2_trials))
        print('*'*20)
        #print(linf_trials)
        print('mean L_inf: %2.3e' % np.mean(linf_trials))
        print('std  L_inf: %2.3e' % np.std(linf_trials))
        print('*'*20)
