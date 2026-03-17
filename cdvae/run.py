from pathlib import Path
from typing import List
import logging
import re

import hydra
import numpy as np
import torch
import omegaconf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from cdvae.common.compat import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    WandbLogger,
    pl,
    seed_everything,
)
from cdvae.common.utils import log_hyperparameters, PROJECT_ROOT


log = logging.getLogger(__name__)


def _select_resume_checkpoint(hydra_dir: Path) -> str | None:
    last_ckpt = hydra_dir / 'last.ckpt'
    if last_ckpt.exists():
        return str(last_ckpt)

    ckpts = list(hydra_dir.glob('*.ckpt'))
    epoch_ckpts: list[tuple[int, int, Path]] = []
    for ckpt in ckpts:
        epoch_match = re.search(r'epoch=(\d+)', ckpt.name)
        step_match = re.search(r'step=(\d+)', ckpt.name)
        if epoch_match is None:
            continue
        epoch = int(epoch_match.group(1))
        step = int(step_match.group(1)) if step_match is not None else -1
        epoch_ckpts.append((epoch, step, ckpt))

    if not epoch_ckpts:
        return None

    _, _, newest_ckpt = max(epoch_ckpts, key=lambda item: (item[0], item[1]))
    return str(newest_ckpt)


def _normalize_trainer_kwargs(cfg: DictConfig, ckpt_path: str | None) -> dict:
    trainer_cfg = OmegaConf.to_container(cfg.train.pl_trainer, resolve=True)
    trainer_kwargs = dict(trainer_cfg)

    gpus = trainer_kwargs.pop("gpus", None)
    if gpus is not None and "devices" not in trainer_kwargs:
        if isinstance(gpus, str):
            if gpus in {"0", "[]", "none", "None"}:
                trainer_kwargs.setdefault("accelerator", "cpu")
                trainer_kwargs["devices"] = 1
            elif "," in gpus:
                gpu_ids = [gpu_id.strip() for gpu_id in gpus.split(",") if gpu_id.strip()]
                trainer_kwargs.setdefault("accelerator", "gpu")
                trainer_kwargs["devices"] = gpu_ids if gpu_ids else 1
            else:
                trainer_kwargs.setdefault("accelerator", "gpu")
                trainer_kwargs["devices"] = int(gpus)
        elif isinstance(gpus, int):
            if gpus <= 0:
                trainer_kwargs.setdefault("accelerator", "cpu")
                trainer_kwargs["devices"] = 1
            else:
                trainer_kwargs.setdefault("accelerator", "gpu")
                trainer_kwargs["devices"] = gpus
        else:
            trainer_kwargs.setdefault("accelerator", "gpu")
            trainer_kwargs["devices"] = gpus

    if "progress_bar_refresh_rate" in trainer_kwargs:
        refresh_rate = trainer_kwargs.pop("progress_bar_refresh_rate")
        trainer_kwargs.setdefault("enable_progress_bar", bool(refresh_rate))

    if ckpt_path is not None:
        trainer_kwargs.setdefault("enable_checkpointing", True)

    trainer_kwargs.setdefault("accelerator", "auto")
    trainer_kwargs.setdefault("devices", 1)
    return trainer_kwargs


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                save_last=cfg.train.model_checkpoints.save_last,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if cfg.train.pl_trainer.fast_dev_run:
        log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        if "gpus" in cfg.train.pl_trainer:
            cfg.train.pl_trainer.gpus = 0
        else:
            cfg.train.pl_trainer.accelerator = "cpu"
            cfg.train.pl_trainer.devices = 1
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpt = _select_resume_checkpoint(hydra_dir)
    if ckpt is not None:
        log.info(f"Found checkpoint: {ckpt}")
    
    trainer_kwargs = _normalize_trainer_kwargs(cfg, ckpt)

    log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        **trainer_kwargs,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    log.info("Starting training")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

    log.info("Starting testing")
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    best_ckpt_path = None
    if checkpoint_callback is not None:
        best_ckpt_path = getattr(checkpoint_callback, "best_model_path", None) or None

    if cfg.train.pl_trainer.fast_dev_run or best_ckpt_path is None:
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.test(datamodule=datamodule, ckpt_path=best_ckpt_path)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
