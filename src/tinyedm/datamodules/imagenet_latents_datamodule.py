import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from .abstract_datamodule import AbstractDataModule


class ImageNetLatentsDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.latents_dir = root / "latents"
        self.labels_dir = root / "labels"

        self.num_files = len(list(self.latents_dir.glob("*.npy")))

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        feature_file = self.latents_dir / f"{idx}.npy"
        label_file = self.labels_dir / f"{idx}.npy"

        feature = np.load(feature_file)
        label = np.load(label_file)

        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label).long()

        return feature, label


class ImageNetLatentsDataModule(AbstractDataModule):
    def __init__(
        self, data_dir, image_size, batch_size, num_workers,
    ):
        super().__init__(data_dir, batch_size, num_workers)
        self.data_dir = Path(data_dir)
        self.image_size = image_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageNetLatentsDataset(self.data_dir / "train",)
            self.val_dataset = ImageNetLatentsDataset(self.data_dir / "val",)

    @property
    def num_classes(self) -> int:
        return 1000
