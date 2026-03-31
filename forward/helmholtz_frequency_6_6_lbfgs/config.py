from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np


@dataclass
class Config:
    # Problem setting
    a: np.ndarray = field(
        default_factory=lambda: np.array([6.0, 6.0], dtype=float)
    )

    # Geometry
    domain_spatio: np.ndarray = field(
        default_factory=lambda: np.array([[-1.0, -1.0],
                                          [ 1.0,  1.0]], dtype=float)
    )

    # Sampling strategy: "uniform" or "random"
    sampling: str = "uniform"
    
    # Uniform mesh sampling
    dom_dis: List[int] = field(default_factory=lambda: [51, 51])
    test_dis: List[int] = field(default_factory=lambda: [201, 201])
    # Random sampling
    # n_dom: int = 2000
    # n_bc: int = 200
    # n_test: int = 10000
    
    # Network
    n_hidden: int = 3
    w_neurons: int = 40
    in_dim: int = 2       # x, y
    out_dim: int = 1      # u
    fourier_features: bool = False
    sigma: float = 1.0

    # Training / logging
    trials: int = 5
    epochs: int = 50_000
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
            f"helm_freq_a{int(self.a[0])}_{int(self.a[1])}"
            f"_nn{self.n_hidden}_{self.w_neurons}"
        )
