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

    

# --------------------------- fully-connected network (x,t) -> (u, f): f=NN(x, 0) ---------------------------

class doubleforward_FC_Net(nn.Module):
    def __init__(self, layers: List[int],
                 mean_dom_st: torch.Tensor, stdev_dom_st: torch.Tensor,
                 fourier_features: bool = False, sigma: float = 1.0):
        super().__init__()
        self.mean = torch.nn.Parameter(mean_dom_st, requires_grad=False)
        self.stdev = torch.nn.Parameter(stdev_dom_st, requires_grad=False)
        
        self.net = make_mlp(layers, final_linear=True,
                            fourier_features=fourier_features, sigma=sigma)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor]:
        data = torch.cat([x, t], dim=1)
        data = (data - self.mean)/self.stdev
        u  = self.net(data)[:,0:1]
        
        data_f = data.clone()
        data_f[:,1:2] = 0.
        f    = self.net(data_f)[:,1:2]
        return u, f
