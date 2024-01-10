from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from .solver import DiffusionSolver, DeterministicSolver
import torch
from torchvision.utils import make_grid
import wandb

class GenerateCallback(Callback):
    def __init__(
        self,
        solver: DiffusionSolver,
        num_samples: int = 8,
        img_shape: tuple[int, int, int] = (3, 32, 32),
        every_n_epochs=5,
    ):
        super().__init__()
        self.solver = solver
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            pl_module.eval()
            with torch.no_grad():
                x0 = torch.randn(
                    self.num_samples, *self.img_shape, device=pl_module.device
                )
                xT = self.solver.solve(pl_module, x0)
                # add to wandblogger
                grid = make_grid(xT, nrow=4, normalize=True, value_range=(-1, 1))
                trainer.logger.log_image(key="generated", images=[grid], step=trainer.current_epoch)

            pl_module.train()

class UploadCheckpointCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer, pl_module):
        best_model_path = trainer.checkpoint_callback.best_model_path
        artifact = wandb.Artifact('checkpoints', type='model')
        artifact.add_file(best_model_path)
        self.wandb_logger.experiment.log_artifact(artifact)

def get_default_callbacks(solver_dtype) -> list[Callback]:
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_summary = ModelSummary(max_depth=1)
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", mode="min", verbose=True
    )
    generate_callback = GenerateCallback(
        DeterministicSolver(dtype=solver_dtype), every_n_epochs=5
    )
    upload_checkpoint_callback = UploadCheckpointCallback()

    default_callbacks = [
        model_summary,
        lr_monitor,
        generate_callback,
        checkpoint_callback, 
        upload_checkpoint_callback,
    ]

    return default_callbacks
