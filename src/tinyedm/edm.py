import lightning as L
import torch
from torch import Tensor, nn, optim
from .metric import WeightedMeanSquaredError
from .networks import Linear

from typing import Protocol


class DiffuserProtocol(Protocol):
    @torch.no_grad()
    def __call__(self, clean_image: Tensor) -> tuple[Tensor, Tensor]:
        ...


class EmbeddingProtocol(Protocol):
    def __call__(self, sigma: Tensor, class_label: Tensor | None = None) -> Tensor:
        ...

    @property
    def embedding_dim(self) -> int:
        ...


class DenoiserProtocol(Protocol):
    def __call__(
        self, noisy_image: Tensor, sigma: Tensor, embedding: Tensor | None = None
    ) -> Tensor:
        ...

    @property
    def sigma_data(self) -> float:
        ...


class EDMDiffuser:
    """
    A diffusion model that adds Gaussian noise to the input. The noise is sampled from

        ln(sigma) ~ N(P_mean, P_std)

    Parameters:
        P_mean: The mean of the log of the noise.
        P_std: The standard deviation of the log of the noise.

    Returns:
        A tuple of (noisy_image, sigma).

    """

    def __init__(self, P_mean, P_std: float) -> None:
        self.P_mean = P_mean
        self.P_std = P_std

    @torch.no_grad()
    def __call__(self, clean_image: Tensor) -> tuple[Tensor, Tensor]:
        epsilon = torch.randn(clean_image.shape[0], device=clean_image.device)
        sigma = (self.P_mean + epsilon * self.P_std).exp()

        noise = torch.randn_like(clean_image)
        noise = noise * sigma.view(-1, 1, 1, 1)
        return clean_image + noise, sigma


class EDMDenoiser(nn.Module):
    """
    A denoiser proposed in [1]. It wraps a neural network with a skip-connection-like structure.

    Parameters:
        net: The neural network.
        sigma_data: The estimated standard deviation of the data.

    Returns:
        The denoised image.

    [1]: Karras T, Aittala M, Aila T, et al. Elucidating the design space of diffusion-based generative models[J].
         Advances in Neural Information Processing Systems, 2022, 35: 26565-26577.


    """

    def __init__(self, net: nn.Module, sigma_data: float):
        super().__init__()

        self.net = net
        self._sigma_data = sigma_data

    @property
    def sigma_data(self) -> float:
        return self._sigma_data

    def forward(
        self, noisy_image: Tensor, sigma: Tensor, embedding: Tensor | None = None
    ) -> Tensor:
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


class EDM(L.LightningModule):
    def __init__(
        self,
        *,
        diffuser: DiffuserProtocol,
        denoiser: DenoiserProtocol,
        embedding: EmbeddingProtocol,
        sigma_data: float,
        use_uncertainty: bool,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        super().__init__()

        self.diffuser = diffuser
        self.denoiser = denoiser
        self.embedding = embedding
        self.u = (
            Linear(embedding.embedding_dim, 1)
            if use_uncertainty
            else lambda x: torch.tensor(0.0)
        )
        self.sigma_data = sigma_data
        self.lr = lr
        self.betas = betas
        self.mse = WeightedMeanSquaredError()

    def training_step(self, batch, batch_idx):
        clean_image, class_label = batch
        noisy_image, sigma = self.diffuser(clean_image)
        embedding = self.embedding(sigma, class_label)
        denoised_image = self.denoiser(noisy_image, sigma, embedding)

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        uncertainty = self.u(embedding)
        loss = (
            self.mse(weight / uncertainty.exp(), denoised_image, clean_image)
            + uncertainty.mean()
        )
        self.log("train_loss", self.mse, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
