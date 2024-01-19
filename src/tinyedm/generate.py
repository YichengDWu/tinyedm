import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tinyedm.datamodule import RandomNoiseDataModule
from tinyedm import PreditionWriter
import lightning as L
from tinyedm import DeterministicSolver
import argparse


def generate(
    ckpt_path,
    config_path,
    output_dir,
    num_samples,
    image_size,
    num_classes,
    batch_size,
    num_workers=16,
    num_steps=32,
) -> None:
    # instantiate model from a yaml config, note if you override some parameters in the command line, instantiation will not reflect that
    yaml_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    cfg = OmegaConf.create(yaml_config)
    model = instantiate(cfg.model)

    model.solver = DeterministicSolver(num_steps=num_steps)

    # noise datamodule
    datamodule = RandomNoiseDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        num_samples=num_samples,
        num_classes=num_classes,
    )

    mean, std = (
        (0.49139968, 0.48215841, 0.44653091),
        (0.24703223, 0.24348513, 0.26158784),
    )  # need to do better
    prediction_writer = PreditionWriter(
        output_dir=output_dir, write_interval="batch", mean=mean, std=std
    )

    trainer = L.Trainer(
        accelerator="gpu",
        strategy="auto",
        callbacks=[prediction_writer],
        enable_model_summary=False,
    )
    trainer.predict(
        model, datamodule=datamodule, return_predictions=False, ckpt_path=ckpt_path
    )


def main():
    parser = argparse.ArgumentParser(description="Run the model generation")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for output"
    )
    parser.add_argument(
        "--num_samples", type=int, required=True, help="Number of samples to generate"
    )
    parser.add_argument("--image_size", type=int, required=True, help="Image size")
    parser.add_argument(
        "--num_classes", type=int, required=True, help="Number of classes"
    )
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers (default: 16)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=32, help="Number of steps (default: 32)"
    )
    args = parser.parse_args()

    # Call generate with arguments from command line
    generate(
        args.ckpt_path,
        args.config_path,
        args.output_dir,
        args.num_samples,
        args.image_size,
        args.num_classes,
        args.batch_size,
        args.num_workers,
        args.num_steps,
    )


if __name__ == "__main__":
    main()