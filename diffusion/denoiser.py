from torch import nn, Tensor
import torch


class Denoiser(nn.Module):
    def __init__(self, net: nn.Module, sigma_data: float = 0.5):
        super().__init__()

        self.net = net
        self.sigma_data = sigma_data

    def forward(self, noisy_image: Tensor, sigma: Tensor) -> Tensor:
        if sigma.ndim == 0:
            sigma = sigma * torch.ones(
                noisy_image.shape[0], dtype=noisy_image.dtype, device=noisy_image.device
            )
        sigma = sigma.view(-1, 1, 1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F = self.net(c_in * noisy_image, c_noise.flatten())
        D = c_skip * noisy_image + c_out * F
        return D
