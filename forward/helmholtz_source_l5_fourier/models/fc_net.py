import torch
import torch.nn as nn

from typing import Tuple, List

from utils import FourierFeatureLayer, NonLinearLayer, InputNormalizer, make_mlp

# -----------------------------
# Model
# -----------------------------


def xavier_init(m: torch.nn.Module):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

        
# --------------------------- spatial network (x,y) -> (u) ---------------------------

class SpatialNet(nn.Module):
    def __init__(self, layers: List[int],
                 mean: torch.Tensor, stdev: torch.Tensor,
                 fourier_features: bool = False, sigma: float = 1.0):
        super().__init__()
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.stdev = torch.nn.Parameter(stdev, requires_grad=False)
        
        self.net = make_mlp(layers, final_linear=True,
                            fourier_features=fourier_features, sigma=sigma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor]:
        data = torch.cat([x, y], dim=1)                
        out = self.net( (data - self.mean)/self.stdev )   
        
        return out

    
'''
# --------------------------- fully-connected network (x,y,z,t) -> (u,v,w,p) ---------------------------

class FC_Net(nn.Module):
    """
    Spatio-temporal network:
    """
    def __init__(self, layers: List[int],
                 mean_dom_st: torch.Tensor, stdev_dom_st: torch.Tensor,
                 fourier_features: bool = False, sigma: float = 1.0):
        super().__init__()
        self.in_norm = InputNormalizer(mean_dom_st, stdev_dom_st)
        self.net = make_mlp(layers, final_linear=True,
                            fourier_features=fourier_features, sigma=sigma)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                z: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor]:
        data = torch.cat([x, y, z, t], dim=1)
        data = self.in_norm(data)

        out = self.net( data )
        u   = out[:, 0:1]
        v   = out[:, 1:2]
        w   = out[:, 2:3]
        p   = out[:, 3:4]
        return u, v, w, p
'''