from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from hora.utils.checkpoint_utils import get_stage_best_checkpoint_relpath
from hora.utils.eval_sweep import infer_output_name_from_checkpoint


def resolve_checkpoint_and_output_name(
    run_name: str | None,
    stage: int,
    checkpoint: str | None = None,
) -> tuple[str, str]:
    if checkpoint:
        return checkpoint, infer_output_name_from_checkpoint(checkpoint)
    if not run_name:
        raise ValueError("Either run_name or checkpoint must be provided")
    output_name = f"AllegroHandHora/{run_name}"
    return get_stage_best_checkpoint_relpath(output_name, stage), output_name


def default_recording_path(
    output_name: str,
    stage: int,
    object_type: str,
    ext: str = "gif",
) -> Path:
    run_name = output_name.split("/")[-1]
    safe_object_type = object_type.replace("+", "_")
    return Path("outputs") / "recordings" / f"{run_name}__stage{stage}__{safe_object_type}.{ext}"


def build_recording_overrides(
    *,
    output_name: str,
    checkpoint: str,
    stage: int,
    object_type: str,
    use_tactile: bool,
    num_envs: int,
    extra_overrides: Iterable[str] = (),
) -> list[str]:
    algo = "PPO" if stage == 1 else "ProprioAdapt"
    overrides = [
        "task=AllegroHandHora",
        "headless=True",
        "pipeline=gpu",
        "test=True",
        "task.on_evaluation=False",
        "task.enableCameraSensors=True",
        f"task.env.numEnvs={num_envs}",
        f"task.env.object.type={object_type}",
        "task.env.randomization.randomizeMass=False",
        "task.env.randomization.randomizeCOM=False",
        "task.env.randomization.randomizeFriction=False",
        "task.env.randomization.randomizePDGains=False",
        "task.env.randomization.randomizeScale=False",
        "task.env.randomization.jointNoiseScale=0.0",
        "task.env.forceScale=0.0",
        "task.env.randomForceProbScalar=0.0",
        f"train.algo={algo}",
        "train.ppo.priv_info=True",
        f"train.ppo.proprio_adapt={'True' if stage == 2 else 'False'}",
        f"task.env.hora.useTactile={'True' if use_tactile else 'False'}",
        f"train.ppo.output_name={output_name}",
        f"checkpoint={checkpoint}",
        *list(extra_overrides),
    ]
    return overrides


def rgba_image_to_rgb_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape == (height, width, 4):
        rgba = array
    elif array.ndim == 2 and array.shape == (height, width * 4):
        rgba = array.reshape(height, width, 4)
    elif array.ndim == 1 and array.size == height * width * 4:
        rgba = array.reshape(height, width, 4)
    else:
        raise ValueError(f"Unsupported camera image shape {array.shape}; expected packed RGBA for {width}x{height}")

    rgb = np.ascontiguousarray(rgba[..., :3])
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb
