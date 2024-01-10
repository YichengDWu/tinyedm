import torch
from torch import Tensor


class Diffuser:
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2) -> None:
        self.P_mean = P_mean
        self.P_std = P_std

    def __call__(self, clean_image: Tensor) -> tuple[Tensor, Tensor]:
        epsilon = torch.randn(clean_image.shape[0], device=clean_image.device)
        sigma = (self.P_mean + epsilon * self.P_std).exp()

        noise = torch.randn_like(clean_image)
        noise = noise * sigma.view(-1, 1, 1, 1)
        return clean_image + noise, sigma
