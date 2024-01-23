from torchvision.datasets import ImageNet
from torchvision.transforms import v2, InterpolationMode
import torch
from .abstract_datamodule import AbstractDataModule
from pathlib import Path
from typing import Sequence


class ImagenetDataModule(AbstractDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__(data_dir, batch_size, num_workers)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    [500, 500], interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                v2.CenterCrop(self.image_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=self.mean,
                    std=map(lambda x: 2 * x, self.std),
                ),
            ]
        )

    def prepare_data(self):
        # verify if there data is already in data_dir
        data_dir = Path(self.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

        expected_dirs = ["train", "val"]
        for expected_dir in expected_dirs:
            if not (data_dir / expected_dir).exists():
                raise FileNotFoundError(
                    f"Data directory {data_dir} does not contain {expected_dir} directory."
                )

            subdirs = [d for d in (data_dir / expected_dir).iterdir() if d.is_dir()]
            if len(subdirs) < 1000:
                raise FileNotFoundError(
                    f"Data directory {data_dir} does not contain 1000 subdirectories."
                )

        print(f"Found data in {data_dir}.")

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageNet(
                self.data_dir,
                split="train",
                transform=self.transforms,
            )
            self.val_dataset = ImageNet(
                self.data_dir,
                split="val",
                transform=self.transforms,
            )
