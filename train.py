import torch
from diffusion.datamodule import CIFAR10DataModule
from diffusion import (
    Diffuser,
    Denoiser,
    EDM,
    get_default_callbacks,
)
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
from diffusers import UNet2DModel
import torch.nn as nn

# Setting the seed
L.seed_everything(42)


class UNetWrapper(nn.Module):
    def __init__(self, model):
        super(UNetWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.sample


def main():
    cifar10 = CIFAR10DataModule(batch_size=128, num_workers=16)
    cifar10.prepare_data()
    cifar10.setup("fit")
    diffuser = Diffuser()

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

    wandb_logger = WandbLogger(project="MNIST", name="bigrun", log_model=True)

    model = EDM(denoiser=denoiser, diffuser=diffuser)
    wandb_logger.watch(model, log_freq=500)
    callbacks = get_default_callbacks()

    trainer = L.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=1000,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=16,
        strategy="ddp",
    )

    trainer.fit(model, cifar10)
    
    
if __name__ == "__main__":
    main()