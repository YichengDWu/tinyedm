from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .abstract_datamodule import AbstractDataModule


class MNISTDataModule(AbstractDataModule):
    def __init__(self, data_dir: str, batch_size: int = 16, num_workers: int = 4):
        super().__init__(data_dir, batch_size, num_workers)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage=str | None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                self.data_dir, train=True, download=False, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                mnist_full, [55000, 5000]
            )

        if stage == "test":
            self.test_dataset = MNIST(
                self.data_dir, train=False, download=False, transform=self.transform
            )
