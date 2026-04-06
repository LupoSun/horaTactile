import os

import wandb
from omegaconf import OmegaConf


def resolve_wandb_mode(env=None) -> str:
    env = os.environ if env is None else env
    explicit_mode = env.get("WANDB_MODE")
    if explicit_mode:
        return explicit_mode
    return "online" if env.get("WANDB_API_KEY") else "offline"


def get_wandb_config(full_config):
    return OmegaConf.to_container(full_config, resolve=True)


def init_wandb_run(full_config, name: str, group: str, project: str = "hora"):
    return wandb.init(
        project=project,
        name=name,
        group=group,
        config=get_wandb_config(full_config),
        mode=resolve_wandb_mode(),
    )
