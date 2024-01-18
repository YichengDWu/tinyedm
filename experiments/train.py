import torch
from tinyedm import (
    EDM,
    GenerateCallback,
    UploadCheckpointCallback,
    LogBestCkptCallback,
)

from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import hydra
import wandb
from tinyedm.ema import EMA
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="mnist")
def main(cfg: DictConfig) -> None:
    # Setting the seed
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")

    diffuser = hydra.utils.instantiate(cfg.diffuser)
    embedding = hydra.utils.instantiate(cfg.embedding)
    denoiser = hydra.utils.instantiate(cfg.denoiser)

    if cfg.compile:
        denoiser = torch.compile(denoiser, fullgraph=True)
        embedding = torch.compile(embedding, fullgraph=True)

    model = EDM(
        denoiser=denoiser,
        diffuser=diffuser,
        embedding=embedding,
        **cfg.model,
    )
    print(model)

    solver = hydra.utils.instantiate(cfg.solver, dtype=torch.float32)

    wandb.init(config=OmegaConf.to_container(cfg, resolve=True), **cfg.wandb)
    wandb.run.log_code(".")
    logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint_callback)
    logckptpath_callback = LogBestCkptCallback()
    generate_callback = GenerateCallback(
        solver=solver,
        enable_ema=cfg.ema.enable,
        std=datamodule.std,
        mean=datamodule.mean,
        value_range=(0, 1),
        **cfg.generate_callback,
    )
    upload_callback = UploadCheckpointCallback()
    callbacks = [
        checkpoint_callback,
        logckptpath_callback,
        generate_callback,
        upload_callback,
    ]

    if cfg.ema.enable:
        ema_callback = EMA(
            ema_length=cfg.ema.ema_length,
            validate_original_weights=cfg.ema.validate_original_weights,
            cpu_offload=cfg.ema.cpu_offload,
            every_n_steps=cfg.ema.every_n_steps,
        )
        callbacks.append(ema_callback)

    trainer = L.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    logger.watch(model, **cfg.wandb_watch)

    ckpt_path = getattr(cfg, "ckpt_path", None)
    # Three cases: 1) resume wandb run, 2) start a new run with ckpt_path, 3) start a new run without ckpt_path
    if not wandb.run.resumed:
        if ckpt_path is not None:
            print("Starting a new run with ckpt_path")
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        else:
            print("Starting a new run without ckpt_path")
            trainer.fit(model, datamodule=datamodule)
    elif ckpt_path is None:
        print("Resuming wandb run with ckpt_path saved in wandb")
        ckpt_path = wandb.run.config.get("best_model_path")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        print("Resuming wandb run with loacl ckpt_path")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
