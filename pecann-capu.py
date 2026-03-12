from dataclasses import dataclass
from typing import Callable, Optional
import torch


class PrimalOptimizer:
    """
    Wraps a single primal optimisation step.

    loss_fn(model) -> (loss, diagnostics_dict)
        * loss        – scalar tensor with grad graph for .backward()
        * diagnostics – dict of *detached* quantities the caller wants back
                        (e.g. objective, constr, loss value)

    Returns the diagnostics dict produced by the *accepted* step.
    """
    
    @staticmethod
    def _scheduler_step(scheduler, loss_val: float):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss_val)
        else:
            scheduler.step()
            
    @staticmethod
    def step(model: torch.nn.Module,
             optim: torch.optim.Optimizer,
             loss_fn: Callable,
             using_lbfgs: bool,
             scheduler: Optional[object] = None) -> dict:
        if using_lbfgs:
            def closure():
                optim.zero_grad(set_to_none=True)
                loss, _ = loss_fn(model)
                loss.backward()    
                return loss
            optim.step(closure)
        else:
            optim.zero_grad(set_to_none=True)
            loss, _ = loss_fn(model)
            loss.backward()
            optim.step()

        _, diag = loss_fn(model, training=False)

        if scheduler is not None and not using_lbfgs:
            PrimalOptimizer._scheduler_step(scheduler, diag['loss'].item())

        return diag
        
        
# ------------------------------------------------------------------ #
#  PECANN-CAPU  (ALM state + dual updates)                           #
# ------------------------------------------------------------------ #
 
@dataclass
class PECANNState:
    Lambda: torch.Tensor
    Mu: torch.Tensor
    Bar_v: torch.Tensor
    previous_loss: torch.Tensor


class PECANNTrainer:
    """
    Generic PECANN-CAPU wrapper.
      eval_fn(model) -> objective, constr
    """
    def __init__(self, eval_fn: Callable, eta: torch.Tensor,
                 omega: float = 0.999, zeta: float = 0.99, eps: float = 1e-16):
        self.eval_fn = eval_fn
        self.eta = eta
        self.omega = omega
        self.zeta = zeta
        self.eps = eps
        
    # ---- ALM loss ------------------------------------------------ #
    @staticmethod
    def augmented_lagrangian(objective: torch.Tensor, constr: torch.Tensor, state: PECANNState):
        return objective + (state.Lambda * constr).sum() + 0.5 * (state.Mu * constr.square()).sum()

    # ---- Primal step --------------------------------------------- #
    def step(self, model, optim, state: PECANNState, using_lbfgs: bool, scheduler = None):
        def loss_fn(mdl, training=True):
            objective, constr = self.eval_fn(mdl, training=training)
            loss = self.augmented_lagrangian(objective, constr, state)
            diag = {
                'objective': objective.detach(),
                'constr':    constr.detach(),
                'loss':      loss.detach(),
            }
            return loss, diag
        diag = PrimalOptimizer.step(
            model, optim, loss_fn, using_lbfgs, scheduler
        )
        return diag['objective'], diag['constr'], diag['loss']
    
    # ---- Dual update --------------------------------------------- #
    @torch.no_grad()
    def update_alm(self, state: PECANNState, constr: torch.Tensor, loss: torch.Tensor):
        state.Bar_v.mul_(self.zeta).add_((1 - self.zeta) * constr.square())

        if loss >= self.omega * state.previous_loss:
            state.Lambda.add_(state.Mu * constr)
            state.Mu.copy_(torch.max(self.eta / torch.sqrt(state.Bar_v + self.eps), state.Mu))

        state.previous_loss = loss
        return state

