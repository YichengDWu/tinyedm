import torch
from tinyedm.datamodule import CIFAR10DataModule
from tinyedm import (
    Diffuser,
    Denoiser,
    EDM,
    GenerateCallback,
    UploadCheckpointCallback,
    LogBestCkptCallback,
)

from pathlib import Path
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from diffusers import UNet2DModel
import torch.nn as nn
import hydra
import wandb


class UNetWrapper(nn.Module):
    def __init__(self, model):
        super(UNetWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.sample


@hydra.main(version_base=None, config_path="conf", config_name="CIFAR10")
def main(cfg) -> None:
    # Setting the seed
    L.seed_everything(cfg.seed)

    cifar10 = CIFAR10DataModule(**cfg.model.train_ds)
    cifar10.prepare_data()
    cifar10.setup("fit")
    diffuser = Diffuser()

    # set up denoiser
    net = UNet2DModel(
        sample_size=32,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(
            64,
            64,
            128,
            128,
            256,
            256,
        ),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    net = UNetWrapper(net)
    denoiser = Denoiser(net)

    model = EDM(denoiser=denoiser, diffuser=diffuser)

    solver_dtype = torch.float64 if cfg.solver.dtype == "float64" else torch.float32
    solver = hydra.utils.instantiate(cfg.solver, dtype=solver_dtype)

    wandb.init(**cfg.wandb)
    logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint_callback)
    logckptpath_callback = LogBestCkptCallback()
    generate_callback = GenerateCallback(solver=solver, **cfg.generate_callback)
    upload_callback = UploadCheckpointCallback()
    callbacks = [checkpoint_callback, logckptpath_callback, generate_callback, upload_callback]

    trainer = L.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    logger.watch(model, log_freq=500)

    if wandb.run.resumed:
        ckpt_path = wandb.run.summary.get("best_model_path")
        trainer.fit(model, datamodule=cifar10, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=cifar10)


if __name__ == "__main__":
    main()
