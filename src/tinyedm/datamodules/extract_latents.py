import lightning as L
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL
import argparse
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import torch


class FeaturePredictionWriter(BasePredictionWriter):
    def __init__(self, feature_dir, label_dir, write_interval="batch"):
        super().__init__(write_interval=write_interval)
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.write_interval = write_interval

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        _, class_labels = batch
        for i, idx in enumerate(batch_indices):
            feature_path = self.feature_dir / f"{idx}.npy"
            label_path = self.label_dir / f"{idx}.npy"
            x = prediction[i].cpu().numpy()
            y = class_labels[i].cpu().numpy()
            np.save(feature_path, x)
            np.save(label_path, y)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


class ImageFeatureExtractor(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")

    def on_predict_start(self):
        self.mean = torch.tensor([5.81, 3.25, 0.12, -2.15], device=self.device).view(
            1, -1, 1, 1
        )
        self.std = (
            torch.tensor([4.17, 4.62, 3.71, 3.28], device=self.device).view(1, -1, 1, 1)
            * 2
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        x = self.vae.encode(x).latent_dist.sample()
        # normalize
        x = (x - self.mean) / self.std
        return x

    def predict_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, self.args.image_size)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        dataset = ImageFolder(self.args.data_dir, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader


def main(args):
    L.seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    feature_dir = out_dir / "latents"
    label_dir = out_dir / "labels"

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    model = ImageFeatureExtractor(args)
    feature_writer = FeaturePredictionWriter(
        feature_dir, label_dir, write_interval="batch"
    )
    trainer = L.Trainer(accelerator="gpu", precision=32, callbacks=[feature_writer])
    trainer.predict(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="latents")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()
    main(args)
