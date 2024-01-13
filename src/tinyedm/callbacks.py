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
        enable_ema: True,
        num_samples: int = 8,
        img_shape: tuple[int, int, int] = (3, 32, 32),
        every_n_epochs=5,
    ):
        super().__init__()
        self.solver = solver
        self.enable_ema = enable_ema
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.every_n_epochs = every_n_epochs

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        use_labels = pl_module.num_classes is not None
        if use_labels:
            # randomly sample labels
            self.class_labels = torch.randint(
                0, pl_module.num_classes, (self.num_samples,), device=pl_module.device
            )
        else:
            self.class_labels = "generated"
        self.x0 = torch.randn(
            self.num_samples, *self.img_shape, device=pl_module.device
        )

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
                grid = make_grid(xT, nrow=4, normalize=True, value_range=(-1, 1))
                trainer.logger.log_image(
                    key=self.class_labels, images=[grid], step=trainer.current_epoch
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