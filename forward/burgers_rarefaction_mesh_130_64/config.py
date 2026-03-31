from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np


@dataclass
class Config:
    # Problem setting

    # Geometry Spatio-Temporal
    domain_st: np.ndarray = field(
        default_factory=lambda: np.array([[-0.5, 0.],
                                          [ 0.5, 1.]], dtype=float)
    )

    # Sampling strategy: "uniform" or "random"
    sampling: str = "uniform"
    
    # Uniform mesh sampling
    dom_dis: List[int] = field(default_factory=lambda: [130, 65])
    test_dis: List[int] = field(default_factory=lambda: [258, 129])
    # Random sampling
    # n_dom: int = 2000
    # n_bc: int = 200
    # n_test: int = 10000
    
    # Network
    n_hidden: int = 3
    w_neurons: int = 20
    in_dim: int = 2       # x, y
    out_dim: int = 1      # u
    fourier_features: bool = False
    sigma: float = 1.0

    # Training / logging
    trials: int = 5
    epochs: int = 10_000
    disp: int = 10
    disp2: int = 500
    print_to_console: bool = True

    # Optimization
    use_scheduler: bool = False # for Adam
    switch_to_lbfgs: bool = False 
    lbfgs_start_epoch: int = 0

    # ALM / CAPU params
    num_lambda: int = 2
    eta_vec: Tuple[float, ...] = (0.01,0.01)

    # Paths
    out_dir: str = "logs"
    pic_dir: str = "pic"

    def make_method_name(self) -> str:
        return (
            f"rarefaction_burgers_mesh_{int(self.dom_dis[0])}_{int(self.dom_dis[1])}"
            f"_nn{self.n_hidden}_{self.w_neurons}"
        )
