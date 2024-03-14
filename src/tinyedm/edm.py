import lightning as L
import torch
from torch import Tensor, nn, optim
from .metric import WeightedMeanSquaredError
from .networks import UncertaintyNet
from torch.optim.lr_scheduler import LambdaLR
from .ema import EMA, EMAOptimizer
from .utils import deinstantiate, swap_tensors
from hydra.utils import instantiate
from typing_extensions import Protocol, Any, Self, cast
import numpy as np
import contextlib
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


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

    embedding_dim: int
    fourier_dim: int
    num_classes: int | None

    def __call__(self, sigma: Tensor, class_label: Tensor | None = None) -> Tensor:
        ...


class EDMDenoiser(Protocol):
    """
    A denoiser that takes in a noisy image, the noise level, and an embedding and outputs a denoised image.

    """

    sigma_data: float

    def __call__(self, noisy_image: Tensor, sigma: Tensor, embedding: Tensor) -> Tensor:
        ...


class EDMSolver(Protocol):
    """
    A solver that takes in a model, a Gaussian noise sampled from the standard normal distribution,
    and an optional class label. It iteratively solves the probability flow ODE and outputs the final
    image.
    """

    def solve(self, model: nn.Module, x0: Tensor, class_label: Tensor | None = None):
        ...


class Diffuser(nn.Module):
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
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std

    @torch.no_grad()
    def forward(self, clean_image: Tensor) -> tuple[Tensor, Tensor]:
        epsilon = torch.randn(
            clean_image.shape[0], device=clean_image.device, dtype=clean_image.dtype
        )
        sigma = (self.P_mean + epsilon * self.P_std).exp()

        noise = torch.randn_like(clean_image)
        noise = noise * sigma.view(-1, 1, 1, 1)
        return clean_image + noise, sigma

    def extra_repr(self) -> str:
        return f"P_mean={self.P_mean}, P_std={self.P_std}"


class EDM(L.LightningModule):
    def __init__(
        self,
        *,
        diffuser: EDMDiffuser,
        embedding: EDMEmbedding,
        denoiser: EDMDenoiser,
        use_ema: bool,
        use_uncertainty: bool,
        steady_steps: int,
        rampup_steps: int,
        scheduler_interval: str,
        sigma_data: float | None = None,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        ema_length: float | None = None,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ) -> None:
        super().__init__()

        assert (
            hasattr(embedding, "fourier_dim") and embedding.fourier_dim is not None
        ), "Embedding must have an fourier_dim attribute."

        if use_ema and ema_length is None:
            raise ValueError("ema_length must be specified when use_ema is True.")

        self.diffuser = diffuser
        self.embedding = embedding
        self.denoiser = denoiser
        self.use_ema = use_ema
        self.use_uncertainty = use_uncertainty
        self.steady_steps = steady_steps
        self.rampup_steps = rampup_steps
        self.scheduler_interval = scheduler_interval
        self.betas = betas
        self.ema_length = ema_length
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

        self.u = (
            UncertaintyNet(embedding.fourier_dim, embedding.fourier_dim)
            if use_uncertainty
            else None
        )
        self.sigma_data = sigma_data if sigma_data is not None else denoiser.sigma_data
        self.lr = lr
        self.betas = betas
        self.train_mse = WeightedMeanSquaredError()
        self.val_mse = WeightedMeanSquaredError()
        self.save_config()

    def save_config(self):
        cfg = deinstantiate(self)
        # update self.hparams with the config
        self.hparams.update(cfg)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        *,
        map_location=None,
        load_ema: bool = False,
        **kwargs: Any,
    ) -> Self:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, **kwargs)
        model = instantiate(checkpoint["hyper_parameters"])
        assert isinstance(model, L.LightningModule)

        if load_ema:
            ema_params = cls.find_ema_weights(checkpoint)
            for param, ema_param in zip(model.parameters(), ema_params):
                swap_tensors(param.data, ema_param)

            print("EMA weights loaded.")
            device = next(
                (t for t in ema_params if isinstance(t, torch.Tensor)), torch.tensor(0)
            ).device
            model.to(device)
        else:
            state_dict = checkpoint["state_dict"]
            if not state_dict:
                rank_zero_warn(
                    f"The state dict in {checkpoint_path!r} contains no parameters."
                )
                return cast(Self, model)

            device = next(
                (t for t in state_dict.values() if isinstance(t, torch.Tensor)),
                torch.tensor(0),
            ).device
            model.to(device)
        return cast(Self, model)

    @staticmethod
    def find_ema_weights(checkpoint: dict):
        try:
            ema_params = checkpoint["optimizer_states"][0]["ema"]
            return ema_params
        except KeyError:
            raise ValueError("EMA weights not found in the checkpoint.")

    def training_step(self, batch, batch_idx):
        clean_image, class_label = batch
        class_label = class_label if self.conditional else None
        noisy_image, sigma = self.diffuser(clean_image)
        fourier_embedding, embedding = self.embedding(sigma, class_label)
        denoised_image = self.denoiser(noisy_image, sigma, embedding)

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if self.u is not None:
            uncertainty = self.u(fourier_embedding).flatten()
            uncertainty_mean = uncertainty.mean()
            loss = (
                self.train_mse(weight / uncertainty.exp(), denoised_image, clean_image)
                + uncertainty_mean
            )
            self.log(
                "train_loss", self.train_mse, prog_bar=True,
            )
            self.log(
                "uncertainty", uncertainty_mean,
            )

        else:
            loss = self.train_mse(weight, denoised_image, clean_image)
            self.log(
                "train_loss", self.train_mse, prog_bar=True,
            )

        self.log(
            "learning_rate", self.lr_schedulers().get_last_lr()[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        clean_image, class_label = batch
        class_label = class_label if self.conditional else None
        noisy_image, sigma = self.diffuser(clean_image)
        fourier_embedding, embedding = self.embedding(sigma, class_label)
        denoised_image = self.denoiser(noisy_image, sigma, embedding)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss = self.val_mse(weight, denoised_image, clean_image)

        self.log("val_loss", self.val_mse)
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
                "interval": self.scheduler_interval,
                "frequency": 1,
            },
        }

    def configure_callbacks(self):
        callbacks = []
        if self.use_ema:
            ema_callback = EMA(
                ema_length=self.ema_length,
                validate_original_weights=self.validate_original_weights,
                cpu_offload=self.cpu_offload,
                every_n_steps=self.every_n_steps,
            )
            callbacks.append(ema_callback)
        return callbacks

    def forward(
        self, noisy_image: Tensor, sigma: Tensor, class_label: Tensor | None = None
    ) -> Tensor:
        class_label = class_label if self.conditional else None
        fourier_embedding, embedding = self.embedding(sigma, class_label)
        denoised_image = self.denoiser(noisy_image, sigma, embedding)
        return denoised_image

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int | None = None
    ):
        x0, class_label = batch
        class_label = class_label if self.conditional else None
        xT = self.solver.solve(self, x0, class_label)

        return xT

    @property
    def num_classes(self) -> int | None:
        return self.embedding.num_classes

    @property
    def conditional(self) -> bool:
        return self.num_classes is not None

    @staticmethod
    def get_lr_scheduler(optimizer, rampup_steps, steady_steps):
        def lr_lambda(current_step):
            if current_step < rampup_steps:
                # Linear ramp up phase
                return 1e-8 + (1.0 - 1e-8) * current_step / rampup_steps
            elif current_step < rampup_steps + steady_steps:
                # Constant phase
                return 1.0
            else:
                # Decay phase
                decay_step = current_step - rampup_steps - steady_steps
                return 1 / np.sqrt(1 + decay_step / steady_steps)

        decay_scheduler = LambdaLR(optimizer, lr_lambda)
        return decay_scheduler

    @contextlib.contextmanager
    def swap_ema_weights(self, trainer: L.Trainer):
        optimizer = trainer.optimizers[0]

        if not (self.use_ema and isinstance(optimizer, EMAOptimizer)):
            raise ValueError("EMA is not used or the optimizer is not an EMAOptimizer.")

        optimizer.switch_main_parameter_weights()
        try:
            yield

        finally:
            optimizer.switch_main_parameter_weights()
