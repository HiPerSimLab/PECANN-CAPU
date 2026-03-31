from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np


@dataclass
class Config:
    # Problem setting
    n: float = 0.05
    
    # Geometry Spatio-Temporal
    domain_st: np.ndarray = field(
        default_factory=lambda: np.array([[0., 0.],
                                          [1., 1.]], dtype=float)
    )

    # Sampling strategy: "uniform" or "random"
    sampling: str = "random"
    
    # Uniform mesh sampling
    #dom_dis: List[int] = field(default_factory=lambda: [66, 33])
    test_dis: List[int] = field(default_factory=lambda: [201, 201])
    # Random sampling
    n_dom: int = 10_000 
    n_bc: int = 128   
    n_ic: int = 128   
    n_data: int = 128 
    
    # Network
    n_hidden: int = 1   
    w_neurons: int = 40 
    in_dim: int = 2       # x, y
    out_dim: int = 2      # u
    fourier_features: bool = False
    sigma: float = 1.0

    # Training / logging
    trials: int = 1      
    epochs: int = 10_000 
    disp: int = 10       
    disp2: int = 100   
    print_to_console: bool = True

    # Optimization
    use_scheduler: bool = False # for Adam
    switch_to_lbfgs: bool = True 
    lbfgs_start_epoch: int = 0

    # ALM / CAPU params
    num_lambda: int = 3
    eta_vec: Tuple[float, ...] = (1.,1.,0.01)

    # Paths
    out_dir: str = "logs"
    pic_dir: str = "pic"

    def make_method_name(self) -> str:
        return (
            f"inv_uns_heat"
            f"_nn{self.n_hidden}_{self.w_neurons}"
        )
