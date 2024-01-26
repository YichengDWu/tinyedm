import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    # Setting the seed
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = hydra.utils.instantiate(cfg.model)
    print(model)

    logger = WandbLogger(**cfg.wandb_logger)
    print("wandb logger created")
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
    trainer = L.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)
    print("Trainer created")
    logger.watch(model, **cfg.wandb_watch)

    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path is not None:
        print("Starting a new run with ckpt_path")
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        print("Starting a new run without ckpt_path")
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
