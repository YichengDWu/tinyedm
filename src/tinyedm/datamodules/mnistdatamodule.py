from torch.utils.data import random_split
from torchvision.transforms import v2
from torchvision.datasets import MNIST
import torch
from .abstract_datamodule import AbstractDataModule


class MNISTDataModule(AbstractDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        image_size: int,
        data_dir: str = "datasets/mnist",
    ):
        super().__init__(data_dir, batch_size, num_workers)

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    (0.5,), (0.5,)
                ),  # normalize to have std of 0.5
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MNIST(
                self.data_dir, train=True, download=False, transform=self.transform
            )
            self.val_dataset = MNIST(
                self.data_dir, train=False, download=False, transform=self.transform
            )
        if stage == "test":
            self.test_dataset = MNIST(
                self.data_dir, train=False, download=False, transform=self.transform
            )

    def denormalize(self, x):
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)