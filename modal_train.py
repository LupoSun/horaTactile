"""
Modal cloud training harness for horaTactile (HORA).

Usage:
    # One-time: populate grasp pose cache on volume
    modal run modal_train.py::setup_cache_remote

    # Train both stages sequentially
    modal run modal_train.py --run-name my_exp

    # Train a single stage
    modal run modal_train.py --run-name my_exp --stage 1
    modal run modal_train.py --run-name my_exp --stage 2

    # Pass extra Hydra overrides
    modal run modal_train.py --run-name my_exp --overrides "task.env.numEnvs=4096 train.ppo.max_agent_steps=1024"
"""

from __future__ import annotations

import os
import shlex
import subprocess
import inspect
from pathlib import Path

import modal
from omegaconf import OmegaConf

from hora.utils.checkpoint_utils import get_stage_best_checkpoint_relpath

APP_NAME = "hora-train"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
CONDA_PYTHON = "/usr/bin/python3"
DEFAULT_GPU = os.environ.get("MODAL_GPU", "A100")
DEFAULT_BASE_IMAGE = os.environ.get("MODAL_BASE_IMAGE", "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04")
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24  # 24 hours
VOLUME_COMMIT_INTERVAL_SECONDS = 300
DEFAULT_TASK_NAME = "AllegroHandHora"
DEFAULT_OUTPUT_PREFIX = "AllegroHandHora"
ISAACGYM_FILE_ID = "1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9"
GRASP_CACHE_FILE_ID = "1xqmCDCiZjl2N7ndGsS_ZvnpViU7PH7a3"
LOCAL_REPO_ROOT = Path(__file__).resolve().parent


def _resolve_task_config_path() -> Path:
    remote_candidate = Path(PROJECT_DIR) / "configs" / "task" / f"{DEFAULT_TASK_NAME}.yaml"
    try:
        if remote_candidate.is_file():
            return remote_candidate
    except OSError:
        pass
    return LOCAL_REPO_ROOT / "configs" / "task" / f"{DEFAULT_TASK_NAME}.yaml"


TASK_CONFIG_PATH = _resolve_task_config_path()
IGNORED_PROJECT_PARTS = {
    "outputs",
    "cache",
    "__pycache__",
    ".git",
    ".venv",
    ".pytest_cache",
    ".codex",
    "isaacgym",
}


def _should_copy_project_path(local_path: str) -> bool:
    path = Path(local_path)
    if any(part in IGNORED_PROJECT_PARTS for part in path.parts):
        return False
    return path.suffix != ".pyc"


volume = modal.Volume.from_name("hora-volume", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def _build_modal_image():
    image_obj = (
        modal.Image.from_registry(DEFAULT_BASE_IMAGE, add_python="3.11")
        .pip_install("omegaconf")
        .apt_install("git", "wget", "unzip", "python3", "python3-pip", "python3-dev")
        .run_commands(
            # Isaac Gym Preview 4 requires Python 3.8, so we keep the actual
            # training environment on Ubuntu 20.04's system interpreter while
            # Modal itself runs on its supported standalone Python.
            "/usr/bin/python3 -m pip install --upgrade pip",
            "/usr/bin/python3 -m pip install "
            "torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 "
            "--extra-index-url https://download.pytorch.org/whl/cu118",
            "/usr/bin/python3 -m pip install gdown",
            # IsaacGym Preview 4.0 from NVIDIA's Google Drive
            f"/usr/bin/python3 -m gdown {ISAACGYM_FILE_ID} -O /tmp/isaac4.tar.gz",
            "tar -xzf /tmp/isaac4.tar.gz -C /opt "
            "&& sed -i 's/dtype=np.float/dtype=float/' /opt/isaacgym/python/isaacgym/torch_utils.py "
            "&& cd /opt/isaacgym/python && /usr/bin/python3 -m pip install -e . "
            "&& rm /tmp/isaac4.tar.gz",
            "/usr/bin/python3 -m pip install hydra-core>=1.1 termcolor omegaconf gym wandb",
        )
    )

    if hasattr(image_obj, "add_local_dir"):
        image_obj = image_obj.add_local_dir(
            ".",
            remote_path=PROJECT_DIR,
            copy=True,
            ignore=lambda path: not _should_copy_project_path(str(path)),
        )
    else:
        project_mount = modal.Mount.from_local_dir(".", condition=_should_copy_project_path)
        image_obj = image_obj.copy_mount(project_mount, remote_path=PROJECT_DIR)

    return image_obj


image = _build_modal_image()

app = modal.App(APP_NAME)

# Forward WANDB_API_KEY if set locally.
function_secrets = []
if os.environ.get("WANDB_API_KEY"):
    function_secrets.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}

if hasattr(image, "env"):
    image = image.env(env)


_APP_FUNCTION_SUPPORTS_ENV = "env" in inspect.signature(app.function).parameters


def _modal_function_kwargs(**kwargs):
    function_kwargs = dict(kwargs)
    if _APP_FUNCTION_SUPPORTS_ENV:
        function_kwargs["env"] = env
    return function_kwargs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_output_name(run_name: str) -> str:
    return f"{DEFAULT_OUTPUT_PREFIX}/{run_name}"


def parse_overrides(overrides: str) -> tuple[str, ...]:
    stripped = overrides.strip()
    if not stripped:
        return ()
    return tuple(shlex.split(stripped))


def expected_cache_files(config_path: Path = TASK_CONFIG_PATH) -> tuple[str, ...]:
    task_config = OmegaConf.load(config_path)
    cache_name = task_config.env.grasp_cache_name
    scales = task_config.env.randomization.randomizeScaleList
    return tuple(
        f"{cache_name}_grasp_50k_s{str(scale).replace('.', '')}.npy"
        for scale in scales
    )


def is_cache_complete(cache_dir: str, config_path: Path = TASK_CONFIG_PATH) -> bool:
    existing_files = {path.name for path in Path(cache_dir).glob("*.npy")}
    return set(expected_cache_files(config_path)).issubset(existing_files)


def get_stage_best_checkpoint_volume_path(run_name: str, stage: int, volume_path: str = VOLUME_PATH) -> str:
    relpath = get_stage_best_checkpoint_relpath(get_output_name(run_name), stage)
    return os.path.join(volume_path, relpath)


def setup_project_symlinks(project_dir: str = PROJECT_DIR, volume_path: str = VOLUME_PATH):
    """Link outputs/ and cache/ inside the project dir to the persistent volume."""
    for name in ("outputs", "cache"):
        vol_dir = os.path.join(volume_path, name)
        proj_link = os.path.join(project_dir, name)
        os.makedirs(vol_dir, exist_ok=True)
        if not os.path.exists(proj_link):
            os.symlink(vol_dir, proj_link)


def check_no_overwrite(run_name: str, stage: int, volume_path: str = VOLUME_PATH):
    """Fail early if a best checkpoint already exists (avoids train.py's interactive input() prompt)."""
    best_path = get_stage_best_checkpoint_volume_path(run_name, stage, volume_path=volume_path)
    if os.path.exists(best_path):
        raise RuntimeError(
            f"Checkpoint already exists at {best_path}. "
            f"Pick a different --run-name or delete the existing run on the volume."
        )


def check_stage1_exists(run_name: str, volume_path: str = VOLUME_PATH):
    """Ensure stage 1 best checkpoint is present before starting stage 2."""
    best_path = get_stage_best_checkpoint_volume_path(run_name, stage=1, volume_path=volume_path)
    if not os.path.exists(best_path):
        raise RuntimeError(
            f"Stage 1 checkpoint not found at {best_path}. "
            f"Run stage 1 first: modal run modal_train.py --run-name {run_name} --stage 1"
        )


def build_stage1_command(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()) -> list[str]:
    return [
        CONDA_PYTHON, "train.py",
        f"task={DEFAULT_TASK_NAME}", "headless=True",
        f"seed={seed}",
        "task.env.forceScale=2", "task.env.randomForceProbScalar=0.25",
        "train.algo=PPO",
        "task.env.object.type=cylinder_default",
        "train.ppo.priv_info=True", "train.ppo.proprio_adapt=False",
        f"train.ppo.output_name={get_output_name(run_name)}",
        *extra_args,
    ]


def build_stage2_command(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()) -> list[str]:
    return [
        CONDA_PYTHON, "train.py",
        f"task={DEFAULT_TASK_NAME}", "headless=True",
        f"seed={seed}",
        "task.env.numEnvs=20000",
        "task.env.forceScale=2", "task.env.randomForceProbScalar=0.25",
        "train.algo=ProprioAdapt",
        "task.env.object.type=cylinder_default",
        "train.ppo.priv_info=True", "train.ppo.proprio_adapt=True",
        f"train.ppo.output_name={get_output_name(run_name)}",
        f"checkpoint={get_stage_best_checkpoint_relpath(get_output_name(run_name), 1)}",
        *extra_args,
    ]


def _run_with_periodic_commits(cmd: list[str]):
    """Run a subprocess, committing the volume periodically to persist checkpoints."""
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR)
    returncode = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(timeout=VOLUME_COMMIT_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        volume.commit()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_requested_stages(run_name: str, seed: int = 0, stage: str = "both", extra_args: tuple[str, ...] = ()):
    if stage not in ("1", "2", "both"):
        raise ValueError(f"Unsupported stage selection: {stage}")

    if stage in ("1", "both"):
        print(f"[hora] Starting stage 1 training: {run_name}")
        train_stage1_remote.remote(run_name, seed, extra_args)

    if stage in ("2", "both"):
        print(f"[hora] Starting stage 2 training: {run_name}")
        train_stage2_remote.remote(run_name, seed, extra_args)

    print(f"[hora] Done. Outputs on volume at /vol/outputs/{get_output_name(run_name)}/")


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=1800,
    image=image,
))
def setup_cache_remote():
    """One-time: download and unzip the grasp pose cache onto the volume."""
    cache_dir = f"{VOLUME_PATH}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    # Check if already populated
    if is_cache_complete(cache_dir):
        print(f"Cache already populated at {cache_dir}, skipping download.")
        return
    subprocess.run(
        [CONDA_PYTHON, "-m", "gdown", GRASP_CACHE_FILE_ID, "-O", "/tmp/data.zip"],
        check=True,
    )
    subprocess.run(["unzip", "-o", "/tmp/data.zip", "-d", cache_dir], check=True)
    os.remove("/tmp/data.zip")
    volume.commit()
    print(f"Cache populated: {os.listdir(cache_dir)}")


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
))
def train_stage1_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1: PPO training with privileged object information."""
    setup_project_symlinks()
    check_no_overwrite(run_name, stage=1)
    cmd = build_stage1_command(run_name, seed=seed, extra_args=extra_args)
    _run_with_periodic_commits(cmd)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
))
def train_stage2_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2: Proprioceptive adaptation. Requires stage 1 checkpoint on volume."""
    setup_project_symlinks()
    check_stage1_exists(run_name)
    check_no_overwrite(run_name, stage=2)
    cmd = build_stage2_command(run_name, seed=seed, extra_args=extra_args)
    _run_with_periodic_commits(cmd)


@app.local_entrypoint()
def main(run_name: str, seed: int = 0, stage: str = "both", overrides: str = ""):
    """
    Train HORA on Modal.

    Args:
        run_name: Name for this training run (used in output paths and wandb).
        seed: Random seed (default: 0).
        stage: Which stage to train — "1", "2", or "both" (default).
        overrides: Extra Hydra overrides passed to train.py.
    """
    run_requested_stages(run_name, seed=seed, stage=stage, extra_args=parse_overrides(overrides))
