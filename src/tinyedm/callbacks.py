from typing import Sequence
from lightning.pytorch.callbacks import Callback, BasePredictionWriter
from .edm import EDMSolver
import torch
from torchvision.utils import make_grid
import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from .ema import EMAOptimizer
from pathlib import Path
from PIL import Image

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


class PreditionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str, mean: Sequence, std: Sequence):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.mean = mean
        self.std = std
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup(self, trainer, pl_module, stage: str):
        self.std = torch.tensor(self.std, device=pl_module.device).view(1, -1, 1, 1)
        self.mean = torch.tensor(self.mean, device=pl_module.device).view(1, -1, 1, 1)
                              
    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        images = prediction * self.std * 2 + self.mean
        images = torch.clamp(images, 0, 1).permute(0, 2, 3, 1) * 255
        images = images.to(torch.uint8).cpu().numpy()
        for batch_index, image in zip(batch_indices, images):
            image = (image * 255).astype("uint8")
            image = Image.fromarray(image)
            image.save(self.output_dir / f"{batch_index}.png")
