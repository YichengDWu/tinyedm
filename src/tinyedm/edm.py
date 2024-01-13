import lightning as L
from torch import Tensor, nn, optim
from .diffuser import Diffuser
from .metric import WeightedMeanSquaredError
from .unet import Linear


class EDM(L.LightningModule):
    def __init__(
        self,
        denoiser: nn.Module,
        diffuser: Diffuser | None = None,
        lr: float = 1e-4,
        weight_decay=1e-3,
        use_uncertainty: bool = False,
    ) -> None:
        super().__init__()

        self.diffuser = diffuser
        if diffuser is None:
            self.diffuser = Diffuser()
        self.denoiser = denoiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigma_data = self.denoiser.sigma_data

        self.use_uncertainty = use_uncertainty
        self.mse = WeightedMeanSquaredError()
        self.save_hyperparameters(
            ignore=["denoiser"]
        )  # denoiser is a nn.Module, so it can't be saved

    def configure_optimizers(self):
        return optim.AdamW(
            self.denoiser.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )  # TODO: add scheduler

    def forward(self, noisy_image: Tensor, sigma: Tensor | float) -> Tensor:
        return self.denoiser(noisy_image, sigma)

    def training_step(self, batch, batch_idx):
        clean_img, _ = batch
        noisy_img, sigma = self.diffuser(clean_img)
        if self.use_uncertainty:
            denoised_img, uncertainty = self.denoiser(noisy_img, sigma)
        else:
            denoised_img = self.denoiser(noisy_img, sigma)

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        if self.use_uncertainty:
            loss = (
                self.mse(weight / uncertainty.exp(), denoised_img, clean_img)
                + uncertainty.mean()
            )
        else:
            loss = self.mse(weight, denoised_img, clean_img)
        self.log("train_loss", self.mse, prog_bar=True, on_epoch=True)
        return loss
