from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np


@dataclass
class Config:
    # Problem setting
    L: np.ndarray = field(
        default_factory=lambda: np.array([5.0], dtype=float)
    )

    # Geometry
    domain_spatio: np.ndarray = field(
        default_factory=lambda: np.array([[ 0.0,  0.0],
                                          [ 1.0,  1.0]], dtype=float)
    )

    # Sampling strategy: "uniform" or "random"
    sampling: str = "random"
    
    # Uniform mesh sampling
    # dom_dis: List[int] = field(default_factory=lambda: [51, 51])
    test_dis: List[int] = field(default_factory=lambda: [361, 361])
    
    # Random sampling
    n_dom: int = 160**2 
    n_bc: int = 160 
    #n_test: int = 10000
    
    # Network
    n_hidden: int = 3
    w_neurons: int = 60
    in_dim: int = 2       # x, y
    out_dim: int = 1      # u
    fourier_features: bool = True
    sigma: float = 1.0

    # Training / logging
    trials: int = 5
    epochs: int = 80_000
    disp: int = 10
    disp2: int = 1_000
    print_to_console: bool = True

    # Optimization
    use_scheduler: bool = False # for Adam (ReduceLROnPlateau)
    switch_to_lbfgs: bool = True 
    lbfgs_start_epoch: int = 0

    # ALM / CAPU params
    num_lambda: int = 1
    eta_vec: Tuple[float, ...] = (1.0,)

    # Paths
    out_dir: str = "logs"
    pic_dir: str = "pic"

    def make_method_name(self) -> str:
        return (
            f"helm_source_l{int(self.L)}_"
            f"fourier_nn{self.n_hidden}_{self.w_neurons}"
        )
