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

    # Select an explicit runtime profile
    modal run modal_train.py --run-name my_exp --runtime-profile a100_probe --stage 1
    modal run modal_train.py --run-name my_exp --runtime-profile a100_compat --stage 1

    # Pass extra Hydra overrides
    modal run modal_train.py --run-name my_exp --overrides "task.env.numEnvs=4096 train.ppo.max_agent_steps=1024"
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import shlex
import subprocess
from pathlib import Path

import modal
from omegaconf import OmegaConf

from hora.utils.checkpoint_utils import get_stage_best_checkpoint_relpath

APP_NAME = "hora-train"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
CONDA_PYTHON = "/usr/bin/python3"
T4_STABLE_PROFILE = "t4_stable"
A100_PROBE_PROFILE = "a100_probe"
A100_COMPAT_PROFILE = "a100_compat"
RUNTIME_PROFILE_CHOICES = (T4_STABLE_PROFILE, A100_PROBE_PROFILE, A100_COMPAT_PROFILE)
DEFAULT_RUNTIME_PROFILE = os.environ.get("MODAL_RUNTIME_PROFILE", T4_STABLE_PROFILE)
DEFAULT_BASE_IMAGE = os.environ.get("MODAL_BASE_IMAGE", "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04")
DEFAULT_COMPAT_BASE_IMAGE = os.environ.get("MODAL_COMPAT_BASE_IMAGE", "nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04")
T4_GPU = os.environ.get("MODAL_T4_GPU", "T4")
A100_PROBE_GPU = os.environ.get("MODAL_A100_GPU", "A100-40GB")
A100_COMPAT_GPU = os.environ.get("MODAL_A100_COMPAT_GPU", A100_PROBE_GPU)
DEFAULT_TORCH_INSTALL = os.environ.get(
    "MODAL_TORCH_INSTALL",
    "torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 "
    "--extra-index-url https://download.pytorch.org/whl/cu118",
)
COMPAT_TORCH_INSTALL = os.environ.get(
    "MODAL_COMPAT_TORCH_INSTALL",
    "torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 "
    "--extra-index-url https://download.pytorch.org/whl/cu117",
)
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
    "__pycache__",
    ".git",
    ".venv",
    ".pytest_cache",
    ".codex",
    "isaacgym",
    "outputs",
    "cache",
}


def _should_copy_project_path(local_path: str) -> bool:
    path = Path(local_path)
    if any(part in IGNORED_PROJECT_PARTS for part in path.parts):
        return False
    return path.suffix != ".pyc"


volume = modal.Volume.from_name("hora-volume", create_if_missing=True)


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    gpu: str
    image: modal.Image
    description: str
    function_env: dict[str, str]


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def _build_modal_image(base_image: str, torch_install: str):
    image_obj = (
        modal.Image.from_registry(base_image, add_python="3.11")
        .pip_install("omegaconf")
        .apt_install("git", "wget", "unzip", "python3", "python3-pip", "python3-dev")
        .run_commands(
            # Isaac Gym Preview 4 requires Python 3.8, so we keep the actual
            # training environment on Ubuntu 20.04's system interpreter while
            # Modal itself runs on its supported standalone Python.
            "/usr/bin/python3 -m pip install --upgrade pip",
            f"/usr/bin/python3 -m pip install {torch_install}",
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

    netrc = Path("~/.netrc").expanduser()
    if netrc.is_file():
        image_obj = image_obj.add_local_file(netrc, remote_path="/root/.netrc", copy=True)

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


app = modal.App(APP_NAME)

# Forward WANDB_API_KEY if set locally.
function_secrets = []
if os.environ.get("WANDB_API_KEY"):
    function_secrets.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
    "WANDB_MODE": "online",
}

stable_image = _build_modal_image(DEFAULT_BASE_IMAGE, DEFAULT_TORCH_INSTALL)
compat_image = _build_modal_image(DEFAULT_COMPAT_BASE_IMAGE, COMPAT_TORCH_INSTALL)

if hasattr(stable_image, "env"):
    stable_image = stable_image.env(env)
if hasattr(compat_image, "env"):
    compat_image = compat_image.env(env)

_APP_FUNCTION_SUPPORTS_ENV = "env" in inspect.signature(app.function).parameters


def _modal_function_kwargs(function_env: dict[str, str] | None = None, **kwargs):
    function_kwargs = dict(kwargs)
    if _APP_FUNCTION_SUPPORTS_ENV:
        merged_env = dict(env)
        if function_env:
            merged_env.update(function_env)
        function_kwargs["env"] = merged_env
    return function_kwargs


RUNTIME_PROFILES = {
    T4_STABLE_PROFILE: RuntimeProfile(
        name=T4_STABLE_PROFILE,
        gpu=T4_GPU,
        image=stable_image,
        description="Stable baseline validated on T4 with the current Modal image.",
        function_env={"HORA_MODAL_RUNTIME_PROFILE": T4_STABLE_PROFILE},
    ),
    A100_PROBE_PROFILE: RuntimeProfile(
        name=A100_PROBE_PROFILE,
        gpu=A100_PROBE_GPU,
        image=stable_image,
        description="Current Modal image on an explicit A100 for compatibility probing.",
        function_env={
            "HORA_MODAL_RUNTIME_PROFILE": A100_PROBE_PROFILE,
            "CUDA_LAUNCH_BLOCKING": "1",
        },
    ),
    A100_COMPAT_PROFILE: RuntimeProfile(
        name=A100_COMPAT_PROFILE,
        gpu=A100_COMPAT_GPU,
        image=compat_image,
        description="A100 profile with a more conservative Torch/CUDA stack.",
        function_env={"HORA_MODAL_RUNTIME_PROFILE": A100_COMPAT_PROFILE},
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_output_name(run_name: str) -> str:
    return f"{DEFAULT_OUTPUT_PREFIX}/{run_name}"


def get_runtime_profile(runtime_profile: str = DEFAULT_RUNTIME_PROFILE) -> RuntimeProfile:
    try:
        return RUNTIME_PROFILES[runtime_profile]
    except KeyError as exc:
        choices = ", ".join(RUNTIME_PROFILE_CHOICES)
        raise ValueError(f"Unsupported runtime profile: {runtime_profile}. Expected one of: {choices}") from exc


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
    import shutil
    for name in ("outputs", "cache"):
        vol_dir = os.path.join(volume_path, name)
        proj_link = os.path.join(project_dir, name)
        os.makedirs(vol_dir, exist_ok=True)
        if os.path.islink(proj_link):
            pass  # already a symlink, nothing to do
        elif os.path.isdir(proj_link):
            shutil.rmtree(proj_link)
            os.symlink(vol_dir, proj_link)
        elif not os.path.exists(proj_link):
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


def emit_runtime_diagnostics(runtime_profile: str):
    profile = get_runtime_profile(runtime_profile)
    print(f"[hora] Runtime profile: {profile.name}")
    print(f"[hora] Requested Modal GPU: {profile.gpu}")
    print(f"[hora] Profile description: {profile.description}")

    diagnostic_commands = [
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        [
            CONDA_PYTHON,
            "-c",
            (
                "import json, torch; "
                "info = {"
                "'torch_version': torch.__version__, "
                "'torch_cuda': torch.version.cuda, "
                "'cuda_available': torch.cuda.is_available(), "
                "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
                "'device_capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None, "
                "'cudnn_version': torch.backends.cudnn.version()"
                "}; "
                "print(json.dumps(info, sort_keys=True))"
            ),
        ],
    ]
    for command in diagnostic_commands:
        try:
            subprocess.run(command, cwd=PROJECT_DIR, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"[hora] Warning: failed to run diagnostic command {command}: {exc}")


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


def _run_stage(stage: int, run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    setup_project_symlinks()
    if stage == 1:
        check_no_overwrite(run_name, stage=1)
        cmd = build_stage1_command(run_name, seed=seed, extra_args=extra_args)
    elif stage == 2:
        check_stage1_exists(run_name)
        check_no_overwrite(run_name, stage=2)
        cmd = build_stage2_command(run_name, seed=seed, extra_args=extra_args)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    _run_with_periodic_commits(cmd)


def get_stage_remote_functions(runtime_profile: str = DEFAULT_RUNTIME_PROFILE):
    get_runtime_profile(runtime_profile)
    if runtime_profile == T4_STABLE_PROFILE:
        return train_stage1_remote, train_stage2_remote
    if runtime_profile == A100_PROBE_PROFILE:
        return train_stage1_a100_probe_remote, train_stage2_a100_probe_remote
    return train_stage1_a100_compat_remote, train_stage2_a100_compat_remote


def run_requested_stages(
    run_name: str,
    seed: int = 0,
    stage: str = "both",
    extra_args: tuple[str, ...] = (),
    runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
):
    if stage not in ("1", "2", "both"):
        raise ValueError(f"Unsupported stage selection: {stage}")
    profile = get_runtime_profile(runtime_profile)
    stage1_remote, stage2_remote = get_stage_remote_functions(profile.name)

    if stage in ("1", "both"):
        print(f"[hora] Starting stage 1 training: {run_name} [{profile.name}]")
        stage1_remote.remote(run_name, seed, extra_args)

    if stage in ("2", "both"):
        print(f"[hora] Starting stage 2 training: {run_name} [{profile.name}]")
        stage2_remote.remote(run_name, seed, extra_args)

    print(f"[hora] Done. Outputs on volume at /vol/outputs/{get_output_name(run_name)}/")


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=1800,
    image=stable_image,
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
    image=RUNTIME_PROFILES[T4_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[T4_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[T4_STABLE_PROFILE].function_env,
))
def train_stage1_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1: PPO training with privileged object information."""
    emit_runtime_diagnostics(T4_STABLE_PROFILE)
    _run_stage(1, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[T4_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[T4_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[T4_STABLE_PROFILE].function_env,
))
def train_stage2_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2: Proprioceptive adaptation. Requires stage 1 checkpoint on volume."""
    emit_runtime_diagnostics(T4_STABLE_PROFILE)
    _run_stage(2, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[A100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_PROBE_PROFILE].function_env,
))
def train_stage1_a100_probe_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1 on the current image with an explicit A100 for diagnostics."""
    emit_runtime_diagnostics(A100_PROBE_PROFILE)
    _run_stage(1, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[A100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_PROBE_PROFILE].function_env,
))
def train_stage2_a100_probe_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2 on the current image with an explicit A100 for diagnostics."""
    emit_runtime_diagnostics(A100_PROBE_PROFILE)
    _run_stage(2, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[A100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_COMPAT_PROFILE].function_env,
))
def train_stage1_a100_compat_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1 on the alternate A100 compatibility image."""
    emit_runtime_diagnostics(A100_COMPAT_PROFILE)
    _run_stage(1, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[A100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_COMPAT_PROFILE].function_env,
))
def train_stage2_a100_compat_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2 on the alternate A100 compatibility image."""
    emit_runtime_diagnostics(A100_COMPAT_PROFILE)
    _run_stage(2, run_name, seed=seed, extra_args=extra_args)


@app.local_entrypoint()
def main(
    run_name: str,
    seed: int = 0,
    stage: str = "both",
    overrides: str = "",
    runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
):
    """
    Train HORA on Modal.

    Args:
        run_name: Name for this training run (used in output paths and wandb).
        seed: Random seed (default: 0).
        stage: Which stage to train — "1", "2", or "both" (default).
        overrides: Extra Hydra overrides passed to train.py.
        runtime_profile: Modal runtime profile. One of t4_stable, a100_probe, a100_compat.
    """
    run_requested_stages(
        run_name,
        seed=seed,
        stage=stage,
        extra_args=parse_overrides(overrides),
        runtime_profile=runtime_profile,
    )
