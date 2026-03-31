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
    modal run modal_train.py --run-name my_exp -- task.env.numEnvs=4096
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "hora-train"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
CONDA_PYTHON = "/opt/conda/bin/python"
DEFAULT_GPU = "A100"
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24  # 24 hours
VOLUME_COMMIT_INTERVAL_SECONDS = 300
NETRC_PATH = Path("~/.netrc").expanduser()

volume = modal.Volume.from_name("hora-volume", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04")
    .apt_install("git", "wget", "unzip")
    .run_commands(
        # Miniconda + Python 3.8
        "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh",
        "bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh",
        "/opt/conda/bin/conda install -y python=3.8 "
        "pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch",
        # IsaacGym Preview 4.0 from NVIDIA's Google Drive
        "/opt/conda/bin/pip install gdown",
        "gdown 1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9 -O /tmp/isaac4.tar.gz",
        "tar -xzf /tmp/isaac4.tar.gz -C /opt "
        "&& cd /opt/isaacgym/python && /opt/conda/bin/pip install -e . "
        "&& rm /tmp/isaac4.tar.gz",
        # Python dependencies
        "/opt/conda/bin/pip install hydra-core>=1.1 termcolor omegaconf gym wandb",
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=["outputs/", "cache/", "__pycache__/", "*.pyc", ".git/", "isaacgym/"],
    )
)

# Copy W&B credentials into the image if available locally.
if NETRC_PATH.is_file():
    image = image.add_local_file(NETRC_PATH, remote_path="/root/.netrc", copy=True)

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_symlinks():
    """Link outputs/ and cache/ inside the project dir to the persistent volume."""
    for name in ("outputs", "cache"):
        vol_dir = f"{VOLUME_PATH}/{name}"
        proj_link = f"{PROJECT_DIR}/{name}"
        os.makedirs(vol_dir, exist_ok=True)
        if not os.path.exists(proj_link):
            os.symlink(vol_dir, proj_link)


def _check_no_overwrite(run_name: str, stage: int):
    """Fail early if a best checkpoint already exists (avoids train.py's interactive input() prompt)."""
    nn_dir = "stage1_nn" if stage == 1 else "stage2_nn"
    best_ext = "pth" if stage == 1 else "ckpt"
    best_name = "best" if stage == 1 else "model_best"
    best_path = f"{VOLUME_PATH}/outputs/AllegroHandHora/{run_name}/{nn_dir}/{best_name}.{best_ext}"
    if os.path.exists(best_path):
        raise RuntimeError(
            f"Checkpoint already exists at {best_path}. "
            f"Pick a different --run-name or delete the existing run on the volume."
        )


def _check_stage1_exists(run_name: str):
    """Ensure stage 1 best checkpoint is present before starting stage 2."""
    best_path = f"{VOLUME_PATH}/outputs/AllegroHandHora/{run_name}/stage1_nn/best.pth"
    if not os.path.exists(best_path):
        raise RuntimeError(
            f"Stage 1 checkpoint not found at {best_path}. "
            f"Run stage 1 first: modal run modal_train.py --run-name {run_name} --stage 1"
        )


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


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=1800,
    env=env,
    image=image,
)
def setup_cache_remote():
    """One-time: download and unzip the grasp pose cache onto the volume."""
    cache_dir = f"{VOLUME_PATH}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    # Check if already populated
    if any(f.endswith(".npy") for f in os.listdir(cache_dir)):
        print(f"Cache already populated at {cache_dir}, skipping download.")
        return
    subprocess.run(
        ["gdown", "1xqmCDCiZjl2N7ndGsS_ZvnpViU7PH7a3", "-O", "/tmp/data.zip"],
        check=True,
    )
    subprocess.run(["unzip", "-o", "/tmp/data.zip", "-d", cache_dir], check=True)
    os.remove("/tmp/data.zip")
    volume.commit()
    print(f"Cache populated: {os.listdir(cache_dir)}")


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
)
def train_stage1_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1: PPO training with privileged object information."""
    _setup_symlinks()
    _check_no_overwrite(run_name, stage=1)
    cmd = [
        CONDA_PYTHON, "train.py",
        "task=AllegroHandHora", "headless=True",
        f"seed={seed}",
        "task.env.forceScale=2", "task.env.randomForceProbScalar=0.25",
        "train.algo=PPO",
        "task.env.object.type=cylinder_default",
        "train.ppo.priv_info=True", "train.ppo.proprio_adapt=False",
        f"train.ppo.output_name=AllegroHandHora/{run_name}",
        *extra_args,
    ]
    _run_with_periodic_commits(cmd)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
)
def train_stage2_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2: Proprioceptive adaptation. Requires stage 1 checkpoint on volume."""
    _setup_symlinks()
    _check_stage1_exists(run_name)
    _check_no_overwrite(run_name, stage=2)
    cmd = [
        CONDA_PYTHON, "train.py",
        "task=AllegroHandHora", "headless=True",
        f"seed={seed}",
        "task.env.numEnvs=20000",
        "task.env.forceScale=2", "task.env.randomForceProbScalar=0.25",
        "train.algo=ProprioAdapt",
        "task.env.object.type=cylinder_default",
        "train.ppo.priv_info=True", "train.ppo.proprio_adapt=True",
        f"train.ppo.output_name=AllegroHandHora/{run_name}",
        f"checkpoint=outputs/AllegroHandHora/{run_name}/stage1_nn/best.pth",
        *extra_args,
    ]
    _run_with_periodic_commits(cmd)


@app.local_entrypoint()
def main(run_name: str, seed: int = 0, stage: str = "both"):
    """
    Train HORA on Modal.

    Args:
        run_name: Name for this training run (used in output paths and wandb).
        seed: Random seed (default: 0).
        stage: Which stage to train — "1", "2", or "both" (default).
    """
    if stage in ("1", "both"):
        print(f"[hora] Starting stage 1 training: {run_name}")
        train_stage1_remote.remote(run_name, seed)

    if stage in ("2", "both"):
        print(f"[hora] Starting stage 2 training: {run_name}")
        train_stage2_remote.remote(run_name, seed)

    print(f"[hora] Done. Outputs on volume at /vol/outputs/AllegroHandHora/{run_name}/")
