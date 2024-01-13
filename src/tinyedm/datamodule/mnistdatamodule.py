from torch.utils.data import random_split
from torchvision.transforms import v2
from torchvision.datasets import MNIST
import torch
from .abstract_datamodule import AbstractDataModule


class MNISTDataModule(AbstractDataModule):
    def __init__(
        self, batch_size: int, num_workers: int, image_size: int, data_dir: str = "datasets/mnist"
    ):
        super().__init__(data_dir, batch_size, num_workers)

        self.transform = v2.Compose(
            [
                v2.Resize(image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage):
        mnist_full = MNIST(
            self.data_dir, train=True, download=False, transform=self.transform
        )
        self.train_dataset, self.val_dataset = random_split(
            mnist_full, [55000, 5000]
        )
        
    @property
    def classes(self):
        return self.train_dataset.classes