from torch.utils.data import DataLoader
from lightning import LightningDataModule
from abc import abstractmethod


class AbstractDataModule(LightningDataModule):
    def __init__(self, data_dir: str | None, batch_size: int, num_workers: int, warmup_batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.warmup_batch_size = warmup_batch_size
        
    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage=None | str):
        pass
    
    @abstractmethod
    def denormalize(self, x):
        pass

    def train_dataloader(self):
        if self.trainer.current_epoch < self.trainer.model.rampup_steps:
            batch_size = self.warmup_batch_size
        else:
            batch_size = self.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    @property
    def num_classes(self) -> int:
        return len(self.train_dataset.classes)

    @property
    def classes(self) -> list:
        return self.train_dataset.classes
