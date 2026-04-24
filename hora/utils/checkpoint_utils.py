from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class StageCheckpointInfo:
    stage: int
    algo: str
    nn_dir: str
    best_name: str
    best_ext: str

    @property
    def best_filename(self) -> str:
        return f"{self.best_name}.{self.best_ext}"


_STAGE_CHECKPOINT_INFO = {
    1: StageCheckpointInfo(stage=1, algo="PPO", nn_dir="stage1_nn", best_name="best", best_ext="pth"),
    2: StageCheckpointInfo(stage=2, algo="ProprioAdapt", nn_dir="stage2_nn", best_name="model_best", best_ext="ckpt"),
    3: StageCheckpointInfo(stage=3, algo="ProprioAdapt", nn_dir="stage3_nn", best_name="model_best", best_ext="ckpt"),
}

_ALGO_TO_STAGE = {
    "PPO": 1,
    "ProprioAdapt": 2,
}


def get_stage_checkpoint_info(stage: int) -> StageCheckpointInfo:
    if stage not in _STAGE_CHECKPOINT_INFO:
        raise ValueError(f"Unsupported training stage: {stage}")
    return _STAGE_CHECKPOINT_INFO[stage]


def get_algo_checkpoint_info(algo: str) -> StageCheckpointInfo:
    if algo not in _ALGO_TO_STAGE:
        raise ValueError(f"Unsupported training algorithm: {algo}")
    return get_stage_checkpoint_info(_ALGO_TO_STAGE[algo])


def get_stage_best_checkpoint_relpath(output_name: str, stage: int) -> str:
    info = get_stage_checkpoint_info(stage)
    return os.path.join("outputs", output_name, info.nn_dir, info.best_filename)


def get_algo_best_checkpoint_relpath(output_name: str, algo: str) -> str:
    info = get_algo_checkpoint_info(algo)
    return os.path.join("outputs", output_name, info.nn_dir, info.best_filename)


def get_configured_best_checkpoint_relpath(output_name: str, algo: str, nn_dir: Optional[str] = None) -> str:
    info = get_algo_checkpoint_info(algo)
    return os.path.join("outputs", output_name, nn_dir or info.nn_dir, info.best_filename)
