from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np


@dataclass
class Config:
    # Problem setting
    T: float = 8.0
    num_win: int = 40
    
    # Geometry Spatio-Temporal
    domain_st: np.ndarray = field(init=False)
    state_t_array: np.ndarray = field(init=False)
    def __post_init__(self):
        self.domain_st = np.array([[0., 0., 0.],
                                   [1., 1., self.T]], dtype=float)
        # non-overlapping windows
        self.state_t_array = np.linspace(0, self.T, self.num_win+1)

    # Sampling strategy: "uniform"
    sampling: str = "uniform"
    
    # Uniform mesh sampling: per subdomain
    dom_dis: List[int] = field(default_factory=lambda: [257, 257, 11]) 
    test_dis: List[int] = field(default_factory=lambda: [513, 513, 21])

    # Network
    n_hidden: int = 6   
    w_neurons: int = 40 
    in_dim: int = 3       # x, y, t
    out_dim: int = 1      # u
    fourier_features: bool = False
    sigma: float = 1.0

    # Training / logging
    trials: int = 1
    epochs: int = 5_000 
    disp: int = 10
    disp2: int = 500    
    print_to_console: bool = True

    # Optimization
    use_scheduler: bool = False # for Adam
    switch_to_lbfgs: bool = True 
    lbfgs_start_epoch: int = 3000

    # ALM / CAPU params
    num_lambda: int = 2
    eta_vec: Tuple[float, ...] = (0.01,0.01)

    # Paths
    out_dir: str = "logs"
    pic_dir: str = "pic"

    def make_method_name(self) -> str:
        return (
            f"reverse_advection_T{int(self.T)}_mesh{int(self.dom_dis[0])}"
            f"_sig_nn{self.n_hidden}_{self.w_neurons}"
        )
