import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="mnist")
def main(cfg: DictConfig) -> None:
    # Setting the seed
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = hydra.utils.instantiate(cfg.model)
    print(model)

    wandb.init(config=OmegaConf.to_container(cfg, resolve=True), **cfg.wandb)
    wandb.run.log_code(".")
    logger = WandbLogger(cfg.wandb_logger)
    
    callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
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
