from lightning.pytorch.callbacks import Callback
from .edm import EDMSolver
import torch
from torchvision.utils import make_grid
import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from .ema import EMAOptimizer


class LogBestCkptCallback(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.logger.log_hyperparams(
            {"best_model_path": trainer.checkpoint_callback.best_model_path}
        )


class GenerateCallback(Callback):
    def __init__(
        self,
        solver: EDMSolver,
        enable_ema: bool,
        img_shape: tuple[int, int, int],
        mean: tuple,
        std: tuple,
        value_range: tuple[float, float] = (0, 1),
        num_samples: int = 8,
        every_n_epochs=5,
    ):
        super().__init__()
        self.solver = solver
        self.enable_ema = enable_ema
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.every_n_epochs = every_n_epochs
        self.value_range = value_range
        self.mean = mean
        self.std = std

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.class_labels = torch.arange(
            0, pl_module.num_classes, device=pl_module.device, dtype=torch.long
        )
        self.x0 = torch.randn(
            self.num_samples * pl_module.num_classes,
            *self.img_shape,
            device=pl_module.device,
        )
        self.class_labels = self.class_labels.repeat(self.num_samples)

        self.std = torch.tensor(self.std, device=pl_module.device).view(1, -1, 1, 1)
        self.mean = torch.tensor(self.mean, device=pl_module.device).view(1, -1, 1, 1)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            pl_module.eval()
            with torch.no_grad():
                if self.enable_ema:
                    opt = [
                        x for x in trainer.optimizers if isinstance(x, EMAOptimizer)
                    ][0]
                    with opt.swap_ema_weights():
                        xT = self.solver.solve(pl_module, self.x0, self.class_labels)
                else:
                    xT = self.solver.solve(pl_module, self.x0, self.class_labels)
                # add to wandblogger
                # unnormalize
                images = xT * self.std * 2 + self.mean
                images = torch.clamp(images, *self.value_range)
                grid = make_grid(images, nrow=pl_module.num_classes, normalize=False)
                trainer.logger.log_image(
                    key="Generated", images=[grid], step=trainer.current_epoch
                )

            pl_module.train()


class UploadCheckpointCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer, pl_module):
        best_model_path = trainer.checkpoint_callback.best_model_path
        artifact = wandb.Artifact("checkpoints", type="model")
        artifact.add_file(best_model_path)
        trainer.logger.experiment.log_artifact(artifact)
