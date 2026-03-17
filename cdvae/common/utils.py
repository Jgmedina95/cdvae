import os
from pathlib import Path
from typing import Optional

import dotenv
from omegaconf import DictConfig, OmegaConf

from cdvae.common.compat import pl


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(
                f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"


# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel()
                                               for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    logger = trainer.logger
    if logger is None:
        return

    # send hparams to all loggers
    logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
os.environ.setdefault("HYDRA_JOBS", str(DEFAULT_PROJECT_ROOT / "hydra_runs"))
os.environ.setdefault("WABDB_DIR", str(DEFAULT_PROJECT_ROOT / "wandb_runs"))

(Path(os.environ["HYDRA_JOBS"]).expanduser()).mkdir(parents=True, exist_ok=True)
(Path(os.environ["WABDB_DIR"]).expanduser()).mkdir(parents=True, exist_ok=True)

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT")).expanduser().resolve()
assert (
    PROJECT_ROOT.exists()
), "The resolved PROJECT_ROOT path does not exist."

os.chdir(PROJECT_ROOT)
