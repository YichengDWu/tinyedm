from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import torch
from .abstract_datamodule import AbstractDataModule


class CIFAR10DataModule(AbstractDataModule):
    def __init__(
        self,
        data_dir: str = "datasets/cifar",
        image_size: int = 32,
        batch_size: int = 16,
        num_workers: int = 16,
    ):
        super().__init__(data_dir, batch_size, num_workers)

        self.img_size = image_size
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(),
                v2.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # normalize to have std of 0.5
            ]
        )

    def denormalize(self, x):
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)
           
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)
        CIFAR10(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10(
                self.data_dir, train=True, download=False, transform=self.transform,
            )
            self.val_dataset = CIFAR10(
                self.data_dir, train=False, download=False, transform=self.transform,
            )
        if stage == "test":
            self.test_dataset = CIFAR10(
                self.data_dir, train=False, download=False, transform=self.transform,
            )
