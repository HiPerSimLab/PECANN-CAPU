import torch
import numpy as np
import torch.nn as nn
import math
from typing import List, Tuple, Optional, Callable


# -----------------------------
# Layers
# -----------------------------

class FourierFeatureLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, sigma: float = 1.0):
        super().__init__()
        self.B = torch.nn.Parameter(
            torch.randn(input_dim, output_dim // 2) * sigma, requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class NonLinearLayer(torch.nn.Module):
    def __init__(self, in_N: int, out_N: int, act: nn.Module):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_N, out_N), act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InputNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, stdev: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("stdev", stdev.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.stdev


def make_mlp(layers: List[int],
             final_linear: bool = True,
             fourier_features: bool = False,
             sigma: float = 1.0,
             actuation: nn.Module = nn.Tanh()) -> nn.Sequential:
    blocks = []
    if fourier_features:
        out_ff = layers[1]
        blocks.append(FourierFeatureLayer(layers[0], out_ff, sigma=sigma))
    else:
        blocks.append(NonLinearLayer(layers[0], layers[1], actuation))

    for i in range(1, len(layers) - 2):
        blocks.append(NonLinearLayer(layers[i], layers[i + 1], actuation))

    if final_linear:
        blocks.append(nn.Linear(layers[-2], layers[-1]))
    else:
        blocks.append(NonLinearLayer(layers[-2], layers[-1], actuation))
    return nn.Sequential(*blocks)


def stats(domain_coords: np.ndarray) -> np.ndarray:
    coords_mean = (domain_coords[1, :] + domain_coords[0, :]) / 2
    coords_std = (domain_coords[1, :] - domain_coords[0, :]) / math.sqrt(12.0)
    return np.vstack((coords_mean, coords_std))