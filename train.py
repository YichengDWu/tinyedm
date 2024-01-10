import torch
from tinyedm.datamodule import CIFAR10DataModule
from tinyedm import (
    Diffuser,
    Denoiser,
    EDM,
    get_default_callbacks,
)
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from diffusers import UNet2DModel
import torch.nn as nn
import argparse

# Setting the seed
L.seed_everything(42)


class UNetWrapper(nn.Module):
    def __init__(self, model):
        super(UNetWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.sample


def main(args):
    cifar10 = CIFAR10DataModule(batch_size=args.batch_size, num_workers=16)
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

    callbacks = get_default_callbacks(solver_dtype=args.solver_dtype)

    ckpt_path = args.resume_from
    project = args.project
    name = args.name
    max_epochs = args.max_epochs

    if ckpt_path is not None:
        import wandb

        wandb.init(id=args.id, resume="must")
        model = EDM.load_from_checkpoint(ckpt_path, denoiser=denoiser)
        wandb_logger = WandbLogger(project=project, name=name, resume="must")
        trainer = L.Trainer(
            accelerator="gpu",
            devices=-1,
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=callbacks,
            accumulate_grad_batches=16,
            strategy="ddp",
            ckpt_path=ckpt_path,
        )
    else:
        model = EDM(denoiser=denoiser, diffuser=diffuser)
        wandb_logger = WandbLogger(project="MNIST", name="bigrun")
        trainer = L.Trainer(
            accelerator="gpu",
            devices=-1,
            max_epochs=max_epochs,
            logger=wandb_logger,
            callbacks=callbacks,
            accumulate_grad_batches=16,
            strategy="ddp",
        )

    wandb_logger.watch(model, log_freq=500)

    trainer.fit(model, cifar10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run diffusion model training with configurable parameters."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--solver_dtype",
        type=str,
        default="float32",
        help="Data type for solver (float32, float64)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=16,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument("--name", type=str, default="bigrun", help="Name of the run")
    parser.add_argument(
        "--project", type=str, default="MNIST", help="Name of the project"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="ID of the training run, required if resuming",
    )

    args = parser.parse_args()

    if args.resume_from is not None and args.id is None:
        raise ValueError(
            "When resuming training, 'id' parameter must also be specified"
        )

    # Convert solver_dtype to torch dtype
    if args.solver_dtype == "float32":
        args.solver_dtype = torch.float32
    elif args.solver_dtype == "float64":
        args.solver_dtype = torch.float64
    else:
        raise ValueError("Unsupported solver_dtype: must be 'float32' or 'float64'")

    main(args)
