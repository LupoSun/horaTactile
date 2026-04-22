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


def stage_wandb_name(name: str, group: str) -> str:
    suffix = {"stage1": "s1", "stage2": "s2", "stage3": "s3"}.get(group)
    if not suffix:
        return name
    return f"{name}_{suffix}"


def init_wandb_run(full_config, name: str, group: str, project: str = "hora"):
    mode = resolve_wandb_mode()
    if mode == "disabled":
        return None
    return wandb.init(
        project=project,
        name=stage_wandb_name(name, group),
        group=group,
        config=get_wandb_config(full_config),
        mode=mode,
    )
