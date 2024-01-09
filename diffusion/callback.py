from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from .solver import DiffusionSolver, DeterministicSolver
import torch


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

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            x0 = torch.randn(self.num_samples, *self.img_shape, device=pl_module.device)
            xT = self.solver.solve(pl_module, x0)
            images = [
                xT[i, ...].permute(1, 2, 0) / 2 + 0.5 for i in range(self.num_samples)
            ]

            # add to wandblogger
            trainer.logger.experiment.log_image(
                "generated", images, trainer.global_step
            )


def get_default_callbacks() -> list[Callback]:
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_summary = ModelSummary(max_depth=1)
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", mode="min", verbose=True
    )
    generate_callback = GenerateCallback(
        DeterministicSolver(dtype=torch.float32), every_n_epochs=5
    )

    default_callbacks = [
        model_summary,
        lr_monitor,
        generate_callback,
        checkpoint_callback,
    ]

    return default_callbacks
