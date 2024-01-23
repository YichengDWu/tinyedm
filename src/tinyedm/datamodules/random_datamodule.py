from torch.utils.data import Dataset
import torch
from .abstract_datamodule import AbstractDataModule


class RandomNoiseDataset(Dataset):
    def __init__(self, num_samples: int, image_size: int, num_classes: int):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (1,))
        return noise, label


class RandomNoiseDataModule(AbstractDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        image_size: int,
        num_samples: int,
        num_classes: int,
    ):
        super().__init__(None, batch_size, num_workers)
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_classes = num_classes

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.predict_dataset = RandomNoiseDataset(
            self.num_samples, self.image_size, self.num_classes
        )
