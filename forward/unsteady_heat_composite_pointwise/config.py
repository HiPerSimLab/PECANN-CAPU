from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np


@dataclass
class Config:
    # Problem setting

    # Geometry Spatio-Temporal
    domain_st: np.ndarray = field(
        default_factory=lambda: np.array([[-1., 0.],
                                          [ 1., 2.]], dtype=float)
    )

    # Sampling strategy: "uniform" or "random"
    sampling: str = "random"
    
    # Uniform mesh sampling
    #dom_dis: List[int] = field(default_factory=lambda: [66, 33])
    test_dis: List[int] = field(default_factory=lambda: [201, 201])
    # Random sampling
    n_dom: int = 8192 
    n_bc: int = 8192 
    n_ic: int = 8192 
    #n_test: int = 10000
    
    # Network
    n_hidden: int = 6   
    w_neurons: int = 60 
    in_dim: int = 2       # x, y
    out_dim: int = 2      # u
    fourier_features: bool = False
    sigma: float = 1.0

    # Training / logging
    trials: int = 5       
    epochs: int = 500_000 
    disp: int = 100       
    disp2: int = 10_000   
    print_to_console: bool = True

    # Optimization
    use_scheduler: bool = True # for Adam
    switch_to_lbfgs: bool = False 
    lbfgs_start_epoch: int = 0

    # ALM: point-wise params
    num_lambda: int = 0
    eta_vec: Tuple[float, ...] = ()
    def __post_init__(self):
        self.num_lambda = self.n_dom + 2 * self.n_bc + self.n_ic
        self.eta_vec = (0.01,) * self.num_lambda

    # Paths
    out_dir: str = "logs"
    pic_dir: str = "pic"

    def make_method_name(self) -> str:
        return (
            f"unsteady_heat_pntw"
            f"_nn{self.n_hidden}_{self.w_neurons}"
        )
