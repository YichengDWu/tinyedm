import lightning as L
import torch
from torch import Tensor, nn, optim
from .metric import WeightedMeanSquaredError
from .networks import Linear, UncertaintyNet
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ConstantLR, SequentialLR

from typing import Protocol
import numpy as np


class EDMDiffuser(Protocol):
    """
    A diffuser is defined as a function that takes in a clean image and outputs a noisy image and
    the noise level used to generate the noisy image.
    """

    @torch.no_grad()
    def __call__(self, clean_image: Tensor) -> tuple[Tensor, Tensor]:
        ...


class EDMEmbedding(Protocol):
    """
    An embedding that takes in a noise level and an **optional** class label (guidance) and outputs an embedding
    that is then fed into the denoiser.
    """

    def __call__(self, sigma: Tensor, class_label: Tensor | None = None) -> Tensor:
        ...

    @property
    def embedding_dim(self) -> int:
        ...

    @property
    def num_classes(self) -> int | None:
        ...


class EDMDenoiser(Protocol):
    """
    A denoiser that takes in a noisy image, the noise level, and an embedding and outputs a denoised image.

    """

    def __call__(self, noisy_image: Tensor, sigma: Tensor, embedding: Tensor) -> Tensor:
        ...

    @property
    def sigma_data(self) -> float:
        ...


class EDMSolver(Protocol):
    """
    A solver that takes in a model, a Gaussian noise sampled from the standard normal distribution,
    and an optional class label. It iteratively solves the probability flow ODE and outputs the final
    image.
    """

    def solve(self, model: nn.Module, x0: Tensor, class_label: Tensor | None = None):
        ...


class Diffuser:
    """
    A diffusion model that adds Gaussian noise to the input. The noise is sampled from

        ln(sigma) ~ N(P_mean, P_std)

    Parameters:
        P_mean: The mean of the log of the noise.
        P_std: The standard deviation of the log of the noise.

    Returns:
        A tuple of (noisy_image, sigma).

    """

    def __init__(self, P_mean: float, P_std: float) -> None:
        self.P_mean = P_mean
        self.P_std = P_std

    @torch.no_grad()
    def __call__(self, clean_image: Tensor) -> tuple[Tensor, Tensor]:
        epsilon = torch.randn(
            clean_image.shape[0], device=clean_image.device, dtype=clean_image.dtype
        )
        sigma = (self.P_mean + epsilon * self.P_std).exp()

        noise = torch.randn_like(clean_image)
        noise = noise * sigma.view(-1, 1, 1, 1)
        return clean_image + noise, sigma


class EDM(L.LightningModule):
    def __init__(
        self,
        *,
        diffuser: EDMDiffuser,
        denoiser: EDMDenoiser,
        embedding: EDMEmbedding,
        use_uncertainty: bool,
        steady_steps: int,
        rampup_steps: int,
        sigma_data: float | None = None,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        super().__init__()

        self.diffuser = diffuser
        self.denoiser = denoiser
        self.embedding = embedding
        self.use_uncertainty = use_uncertainty
        self.steady_steps = steady_steps
        self.rampup_steps = rampup_steps

        assert (
            hasattr(self.embedding, "embedding_dim")
            and self.embedding.embedding_dim is not None
        ), "Embedding must have an embedding_dim attribute."
        self.u = (
            UncertaintyNet(embedding.embedding_dim, embedding.embedding_dim)
            if use_uncertainty
            else None
        )
        self.sigma_data = sigma_data if sigma_data is not None else denoiser.sigma_data
        self.lr = lr
        self.betas = betas
        self.mse = WeightedMeanSquaredError()

    def training_step(self, batch, batch_idx):
        clean_image, class_label = batch
        noisy_image, sigma = self.diffuser(clean_image)
        embedding = self.embedding(sigma, class_label)
        denoised_image = self.denoiser(noisy_image, sigma, embedding)

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        if self.u is not None:
            uncertainty = self.u(embedding.detach()).flatten()
            loss = (
                self.mse(weight / uncertainty.exp(), denoised_image, clean_image)
                + uncertainty.mean()
            )
        else:
            loss = self.mse(weight, denoised_image, clean_image)

        self.log("train_loss", self.mse, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "learning_rate",
            self.lr_schedulers().get_last_lr()[0],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, betas=self.betas, fused=True
        )

        lr_scheduler = self.get_lr_scheduler(
            optimizer, self.rampup_steps, self.steady_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(
        self, noisy_image: Tensor, sigma: Tensor, class_label: Tensor | None = None
    ) -> Tensor:
        embedding = self.embedding(sigma, class_label)
        denoised_image = self.denoiser(noisy_image, sigma, embedding)
        return denoised_image

    @property
    def num_classes(self) -> int | None:
        return self.embedding.num_classes

    @staticmethod
    def get_lr_scheduler(optimizer, rampup_steps, steady_steps):
        rampup_scheduler = LinearLR(
            optimizer, start_factor=1e-8, total_iters=rampup_steps
        )
        constant_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=steady_steps)

        def lr_lambda(current_step):
            return 1 / np.sqrt(1 + current_step / steady_steps, dtype=np.float32)

        decay_scheduler = LambdaLR(optimizer, lr_lambda)

        return SequentialLR(
            optimizer,
            schedulers=[rampup_scheduler, constant_scheduler, decay_scheduler],
            milestones=[rampup_steps, steady_steps+rampup_steps],
        )
