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

    
