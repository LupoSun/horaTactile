"""
Modal cloud training harness for horaTactile (HORA).

Usage:
    # One-time: populate grasp pose cache on volume
    modal run modal_train.py::setup_cache_remote

    # Train both stages sequentially
    modal run modal_train.py::main --run-name my_exp

    # Train a single stage
    modal run modal_train.py::main --run-name my_exp --stage 1
    modal run modal_train.py::main --run-name my_exp --stage 2
    modal run modal_train.py::main --run-name my_exp --stage 3

    # Train baseline stage 1 + tactile stage 2
    modal run modal_train.py::main --run-name my_exp --runtime-profile h100_stable --tactile

    # Train stages 1/2, then automatically evaluate Stage 2 on BTG1-BTG13 mean objects
    modal run --detach modal_train.py::main --run-name my_exp --runtime-profile a100_compat --stage both --tactile --auto-eval

    # Select an explicit runtime profile
    modal run modal_train.py::main --run-name my_exp --runtime-profile h100_stable --stage 1
    modal run modal_train.py::main --run-name my_exp --runtime-profile a100_probe --stage 1
    modal run modal_train.py::main --run-name my_exp --runtime-profile a100_compat --stage 1
    modal run modal_train.py::main --run-name my_exp --runtime-profile h100_probe --stage 1
    modal run modal_train.py::main --run-name my_exp --runtime-profile h100_compat --stage 1

    # Pass extra Hydra overrides
    modal run modal_train.py::main --run-name my_exp --overrides "task.env.numEnvs=4096 train.ppo.max_agent_steps=1024"

    # Compare baseline Stage 2 vs tactile-enabled Stage 2
    modal run modal_train.py::main --run-name baseline --runtime-profile h100_stable --stage 2
    modal run modal_train.py::main --run-name tactile --runtime-profile h100_stable --stage 2 --tactile
    # Experiment: Run comparison between naively concatenating contact-force tactile signal v.s. baseline in stage two.
    # Note: stage 1 checkpoint best.pth should be uploaded first using modal volume put hora-volume 
    # e.g. modal volume put hora-volume /Users/hz9/dev/horaTactile/outputs/AllegroHandHora/hora_v0.0.2/stage1_nn/best.pth /outputs/AllegroHandHora/double_tactile/stage1_nn/best.pth
    modal run modal_train.py::main --run-name baseline --runtime-profile a100_compat --stage 2
    modal run modal_train.py::main --run-name naive_tactile --runtime-profile a100_compat --stage 2 --overrides "task.env.hora.useTactileHist=True"

    conda activate hora2
    wandb online
    modal run --detach modal_train.py::main --run-name baseline_s1 --runtime-profile a100_compat --stage 1 
    modal run --detach modal_train.py::main --run-name double_tactile_s1 --runtime-profile a100_compat --stage 1 --overrides "task.env.hora.useTactileObs=True"
    modal run --detach modal_train.py::main --run-name double_tactile_s2 --runtime-profile a100_compat --stage 2 --overrides "task.env.hora.useTactileHist=True task.env.hora.useTactileObs=True"
    modal run --detach modal_train.py::main --run-name double_tactile_s2 --runtime-profile a100_compat --stage 3 --overrides "task.env.hora.useTactileHist=True task.env.hora.useTactileObs=True"

    # Run an evaluation sweep on Modal
    modal run --detach modal_train.py::eval_sweep \
        --manifest configs/eval_sweeps/btg13_tactile04201119_10seeds.json \
        --runtime-profile a100_compat

"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
import shlex
import subprocess
from pathlib import Path

import modal
from omegaconf import OmegaConf

from hora.utils.checkpoint_utils import get_stage_best_checkpoint_relpath


def _resolve_modal_gpu_name(preferred: str, fallback: str | None = None) -> str:
    parse_gpu_config = getattr(getattr(modal, "gpu", None), "parse_gpu_config", None)
    if parse_gpu_config is None:
        return preferred
    try:
        parse_gpu_config(preferred)
        return preferred
    except Exception:
        return fallback or preferred


APP_NAME = "hora-train"
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
CONDA_PYTHON = "/usr/bin/python3"
T4_STABLE_PROFILE = "t4_stable"
A100_PROBE_PROFILE = "a100_probe"
A100_COMPAT_PROFILE = "a100_compat"
H100_STABLE_PROFILE = "h100_stable"
H100_PROBE_PROFILE = "h100_probe"
H100_COMPAT_PROFILE = "h100_compat"
RUNTIME_PROFILE_CHOICES = (
    T4_STABLE_PROFILE,
    A100_PROBE_PROFILE,
    A100_COMPAT_PROFILE,
    H100_STABLE_PROFILE,
    H100_PROBE_PROFILE,
    H100_COMPAT_PROFILE,
)
DEFAULT_RUNTIME_PROFILE = os.environ.get("MODAL_RUNTIME_PROFILE", T4_STABLE_PROFILE)
DEFAULT_BASE_IMAGE = os.environ.get("MODAL_BASE_IMAGE", "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04")
DEFAULT_COMPAT_BASE_IMAGE = os.environ.get("MODAL_COMPAT_BASE_IMAGE", "nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04")
T4_GPU = os.environ.get("MODAL_T4_GPU", "T4")
A100_PROBE_GPU = os.environ.get("MODAL_A100_GPU", "A100-40GB")
A100_COMPAT_GPU = os.environ.get("MODAL_A100_COMPAT_GPU", A100_PROBE_GPU)
H100_STABLE_GPU = _resolve_modal_gpu_name(os.environ.get("MODAL_H100_STABLE_GPU", "H100!"), fallback="H100")
H100_PROBE_GPU = _resolve_modal_gpu_name(os.environ.get("MODAL_H100_GPU", "H100!"), fallback="H100")
H100_COMPAT_GPU = _resolve_modal_gpu_name(os.environ.get("MODAL_H100_COMPAT_GPU", H100_PROBE_GPU), fallback="H100")
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
TACTILE_CYLINDER_OBJECT_TYPE = "cylinder_default+custom_cylinder_2dcross+custom_cylinder_3dcross"
TACTILE_CYLINDER_SAMPLE_PROB = "[0.34,0.33,0.33]"
POINTCLOUD_POINT_CHOICES = (100, 200, 300, 500, 1024)
DEFAULT_AUTO_EVAL_NUM_SEEDS = 5
AUTO_EVAL_OBJECT_INDICES = tuple(range(1, 14))
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
        .pip_install("omegaconf", "numpy", "trimesh", "scipy", "matplotlib")
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
            "/usr/bin/python3 -m pip install 'hydra-core>=1.1' termcolor omegaconf gym wandb numpy trimesh scipy matplotlib",
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


app = modal.App(APP_NAME)

# Forward WANDB_API_KEY if set locally.
function_secrets = []
if os.environ.get("WANDB_API_KEY"):
    function_secrets.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
    "MPLCONFIGDIR": "/tmp/matplotlib",
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
    H100_STABLE_PROFILE: RuntimeProfile(
        name=H100_STABLE_PROFILE,
        gpu=H100_STABLE_GPU,
        image=stable_image,
        description="Stable Hopper path validated on H100 with the current Modal image.",
        function_env={"HORA_MODAL_RUNTIME_PROFILE": H100_STABLE_PROFILE},
    ),
    H100_PROBE_PROFILE: RuntimeProfile(
        name=H100_PROBE_PROFILE,
        gpu=H100_PROBE_GPU,
        image=stable_image,
        description="Current Modal image on an explicit H100 for compatibility probing.",
        function_env={
            "HORA_MODAL_RUNTIME_PROFILE": H100_PROBE_PROFILE,
            "CUDA_LAUNCH_BLOCKING": "1",
        },
    ),
    H100_COMPAT_PROFILE: RuntimeProfile(
        name=H100_COMPAT_PROFILE,
        gpu=H100_COMPAT_GPU,
        image=compat_image,
        description="H100 profile with a more conservative Torch/CUDA stack.",
        function_env={"HORA_MODAL_RUNTIME_PROFILE": H100_COMPAT_PROFILE},
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


def with_tactile_overrides(extra_args: tuple[str, ...], tactile: bool = False) -> tuple[str, ...]:
    if not tactile:
        return extra_args
    overrides = list(extra_args)
    if not any(arg.startswith("task.env.hora.useTactileObs=") for arg in overrides):
        overrides.append("task.env.hora.useTactileObs=False")
    if not any(arg.startswith("task.env.hora.useTactileHist=") for arg in overrides):
        overrides.append("task.env.hora.useTactileHist=True")
    return tuple(overrides)


def with_pointcloud_overrides(extra_args: tuple[str, ...], pointcloud_points: int = 1024) -> tuple[str, ...]:
    if pointcloud_points not in POINTCLOUD_POINT_CHOICES:
        raise ValueError(f"Unsupported pointcloud point count: {pointcloud_points}")
    overrides = list(extra_args)
    if not any(arg.startswith("task.env.hora.nPointCloudPts=") for arg in overrides):
        overrides.append(f"task.env.hora.nPointCloudPts={pointcloud_points}")
    if not any(arg.startswith("train.ppo.n_pointcloud_pts=") for arg in overrides):
        overrides.append(f"train.ppo.n_pointcloud_pts={pointcloud_points}")
    return tuple(overrides)


def _override_value(extra_args: tuple[str, ...], key: str) -> str | None:
    prefix = f"{key}="
    for arg in reversed(extra_args):
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _override_bool(extra_args: tuple[str, ...], key: str, default: bool = False) -> bool:
    value = _override_value(extra_args, key)
    if value is None:
        return default
    return value.lower() == "true"


def _pointcloud_eval_overrides(pointcloud_points: int) -> list[str]:
    return [
        f"task.env.hora.nPointCloudPts={pointcloud_points}",
        f"train.ppo.n_pointcloud_pts={pointcloud_points}",
    ]


def build_auto_eval_manifest(
    run_name: str,
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
) -> dict:
    if num_seeds < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
    return {
        "description": (
            "Auto eval for the shape-aware tactile Stage 2 policy on the smallest "
            f"mean-scaled BTG1-BTG13 objects, {num_seeds} seeds per object."
        ),
        "num_envs": 4096,
        "max_evaluate_envs": 20000,
        "seeds": list(range(num_seeds)),
        "base_overrides": [
            "task.env.baseObjScale=1.0",
            "task.env.randomization.graspInitScale=0.8",
            "task.env.reset_height_threshold=0.6",
        ],
        "models": [
            {
                "name": "stage2",
                "checkpoint": get_stage_best_checkpoint_relpath(get_output_name(run_name), 2),
                "algo": "ProprioAdapt",
                "use_tactile_obs": _override_bool(tactile_args, "task.env.hora.useTactileObs", default=False),
                "use_tactile_hist": _override_bool(tactile_args, "task.env.hora.useTactileHist", default=False),
                "use_shape_priv_info": True,
                "env_use_shape_priv_info": False,
                "use_extended_priv_info": True,
                "priv_info_dim": 17,
                "priv_info": True,
                "proprio_adapt": True,
                "output_name": get_output_name(run_name),
                "extra_overrides": _pointcloud_eval_overrides(pointcloud_points),
            }
        ],
        "objects": [
            {
                "name": f"btg{index}_mean",
                "object_type": f"custom_btg{index}_mean",
                "object_index": index,
            }
            for index in AUTO_EVAL_OBJECT_INDICES
        ],
    }


def write_auto_eval_manifest(
    run_name: str,
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
    volume_path: str = VOLUME_PATH,
) -> str:
    manifest = build_auto_eval_manifest(
        run_name,
        tactile_args=tactile_args,
        pointcloud_points=pointcloud_points,
        num_seeds=num_seeds,
    )
    manifest_dir = Path(volume_path) / "eval_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_name}_stage2_btg_mean_{num_seeds}seeds.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return str(manifest_path)


def get_auto_eval_output_dir(run_name: str, volume_path: str = VOLUME_PATH) -> str:
    return str(Path(volume_path) / "outputs" / DEFAULT_OUTPUT_PREFIX / run_name / "stage2_eval")


def get_auto_eval_wandb_name(run_name: str) -> str:
    return f"{get_output_name(run_name)}_eval"


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
            f"Run stage 1 first: modal run modal_train.py::main --run-name {run_name} --stage 1"
        )


def check_stage2_exists(run_name: str, volume_path: str = VOLUME_PATH):
    """Ensure stage 2 best checkpoint is present before starting stage 3."""
    best_path = get_stage_best_checkpoint_volume_path(run_name, stage=2, volume_path=volume_path)
    if not os.path.exists(best_path):
        raise RuntimeError(
            f"Stage 2 checkpoint not found at {best_path}. "
            f"Run stage 2 first: modal run modal_train.py::main --run-name {run_name} --stage 2"
        )


def build_stage1_command(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()) -> list[str]:
    return [
        CONDA_PYTHON, "train.py",
        f"task={DEFAULT_TASK_NAME}", "headless=True",
        f"seed={seed}",
        "task.env.forceScale=2", "task.env.randomForceProbScalar=0.25",
        "train.algo=PPO",
        f"task.env.object.type={TACTILE_CYLINDER_OBJECT_TYPE}",
        f"task.env.object.sampleProb={TACTILE_CYLINDER_SAMPLE_PROB}",
        "task.env.hora.useShapePrivInfo=True",
        "task.env.hora.useExtendedPrivInfo=True",
        "task.env.hora.privInfoDim=17",
        "train.ppo.use_shape_priv_info=True",
        "train.ppo.priv_info_dim=17",
        "train.ppo.priv_info=True", "train.ppo.proprio_adapt=False",
        f"train.ppo.output_name={get_output_name(run_name)}",
        *extra_args,
    ]


def build_stage2_command(
    run_name: str,
    seed: int = 0,
    extra_args: tuple[str, ...] = (),
    tactile: bool = False,
) -> list[str]:
    return [
        CONDA_PYTHON, "train.py",
        f"task={DEFAULT_TASK_NAME}", "headless=True",
        f"seed={seed}",
        "task.env.numEnvs=20000",
        "task.env.forceScale=2", "task.env.randomForceProbScalar=0.25",
        "train.algo=ProprioAdapt",
        f"task.env.object.type={TACTILE_CYLINDER_OBJECT_TYPE}",
        f"task.env.object.sampleProb={TACTILE_CYLINDER_SAMPLE_PROB}",
        "task.env.hora.useShapePrivInfo=True",
        "task.env.hora.useExtendedPrivInfo=True",
        "task.env.hora.privInfoDim=17",
        "train.ppo.use_shape_priv_info=True",
        "train.ppo.priv_info_dim=17",
        "train.ppo.priv_info=True", "train.ppo.proprio_adapt=True",
        f"train.ppo.output_name={get_output_name(run_name)}",
        f"checkpoint={get_stage_best_checkpoint_relpath(get_output_name(run_name), 1)}",
        *with_tactile_overrides(extra_args, tactile=tactile),
    ]


def build_stage3_command(
    run_name: str,
    seed: int = 0,
    extra_args: tuple[str, ...] = (),
    object_type: str = "custom_btg13_mean",
) -> list[str]:
    return [
        CONDA_PYTHON, "train.py",
        f"task={DEFAULT_TASK_NAME}", "headless=True",
        f"seed={seed}",
        "task.env.numEnvs=4096",
        "task.env.forceScale=0.0", "task.env.randomForceProbScalar=0.0",
        "task.env.randomization.randomizeScale=False",
        "task.env.randomization.jointNoiseScale=0.0",
        "task.env.baseObjScale=1.0",
        "task.env.randomization.graspInitScale=0.8",
        "task.env.reset_height_threshold=0.6",
        "train.algo=ProprioAdapt",
        f"task.env.object.type={object_type}",
        "train.ppo.priv_info=True", "train.ppo.proprio_adapt=True",
        "train.ppo.nn_dir=stage3_nn",
        "train.ppo.wandb_group=stage3",
        f"train.ppo.output_name={get_output_name(run_name)}",
        f"checkpoint={get_stage_best_checkpoint_relpath(get_output_name(run_name), 2)}",
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


def build_eval_sweep_command(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
) -> list[str]:
    cmd = [
        CONDA_PYTHON,
        "scripts/eval_object_sweep.py",
        manifest,
        "--python",
        CONDA_PYTHON,
    ]
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    if dry_run:
        cmd.append("--dry-run")
    if wandb_name:
        cmd.extend(["--wandb-name", wandb_name])
        cmd.extend(["--wandb-group", wandb_group])
    return cmd


def _load_mesh_as_trimesh(mesh_path: Path):
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, trimesh.Scene):
        meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unexpected mesh type from {mesh_path}: {type(mesh)}")


def _farthest_point_sample(points, n_points: int):
    import numpy as np

    selected = [0]
    distances = np.full(len(points), np.inf, dtype=np.float64)
    for _ in range(n_points - 1):
        last = points[selected[-1]]
        dist_to_last = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected.append(int(np.argmax(distances)))
    return points[np.asarray(selected)].copy()


def _generate_mesh_pointcloud_sidecar(asset_file: str, n_points: int):
    import numpy as np
    import trimesh

    asset_path = Path(PROJECT_DIR) / asset_file
    visual_path = asset_path.parent / "visual.obj"
    if not visual_path.is_file():
        raise FileNotFoundError(f"Cannot generate point cloud for {asset_file}; missing {visual_path}")

    mesh = _load_mesh_as_trimesh(visual_path)
    mesh.vertices -= mesh.vertices.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
    if radius <= 1e-10:
        raise ValueError(f"Cannot generate point cloud for degenerate mesh {visual_path}")
    mesh.vertices /= radius

    dense_count = max(n_points * 4, n_points)
    sampled, _ = trimesh.sample.sample_surface(mesh, dense_count)
    sampled = np.asarray(sampled, dtype=np.float32)
    points = _farthest_point_sample(sampled, n_points).astype(np.float32)
    output_path = asset_path.parent / f"pointcloud_{n_points}.npy"
    np.save(output_path, points)
    print(f"[hora] Generated missing point cloud sidecar: {output_path.relative_to(Path(PROJECT_DIR))}")


def _ensure_eval_pointcloud_sidecars(manifest: str):
    from hora.utils.object_assets import build_object_asset_catalog, load_object_point_cloud

    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path(PROJECT_DIR) / manifest_path
    data = json.loads(manifest_path.read_text())
    if not any(model.get("env_use_shape_priv_info", model.get("use_shape_priv_info", False)) for model in data.get("models", [])):
        return

    n_points = 1024
    for model in data.get("models", []):
        for override in model.get("extra_overrides", []):
            if override.startswith("task.env.hora.nPointCloudPts="):
                n_points = int(override.split("=", 1)[1])

    asset_files: set[str] = set()
    for obj in data.get("objects", []):
        object_type_list, _, asset_files_dict = build_object_asset_catalog(
            obj["object_type"],
            [1.0],
            repo_root=Path(PROJECT_DIR),
        )
        asset_files.update(asset_files_dict[object_type] for object_type in object_type_list)

    for asset_file in sorted(asset_files):
        try:
            load_object_point_cloud(asset_file, n_points, repo_root=Path(PROJECT_DIR))
        except FileNotFoundError:
            _generate_mesh_pointcloud_sidecar(asset_file, n_points)


def _run_eval_sweep(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
    auto_run_name: str = "",
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    setup_project_symlinks()
    if auto_run_name:
        manifest = write_auto_eval_manifest(
            auto_run_name,
            tactile_args=tactile_args,
            pointcloud_points=pointcloud_points,
            num_seeds=num_seeds,
        )
    if not dry_run:
        _ensure_eval_pointcloud_sidecars(manifest)
    cmd = build_eval_sweep_command(
        manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
    )
    _run_with_periodic_commits(cmd)


def _run_stage(stage: int, run_name: str, seed: int = 0, extra_args: tuple[str, ...] = (), tactile: bool = False):
    setup_project_symlinks()
    if stage == 1:
        check_no_overwrite(run_name, stage=1)
        cmd = build_stage1_command(run_name, seed=seed, extra_args=extra_args)
    elif stage == 2:
        check_stage1_exists(run_name)
        check_no_overwrite(run_name, stage=2)
        cmd = build_stage2_command(run_name, seed=seed, extra_args=extra_args, tactile=tactile)
    elif stage == 3:
        check_stage2_exists(run_name)
        check_no_overwrite(run_name, stage=3)
        cmd = build_stage3_command(run_name, seed=seed, extra_args=extra_args)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    _run_with_periodic_commits(cmd)


def get_stage_remote_functions(runtime_profile: str = DEFAULT_RUNTIME_PROFILE):
    get_runtime_profile(runtime_profile)
    if runtime_profile == T4_STABLE_PROFILE:
        return train_stage1_remote, train_stage2_remote, train_stage3_remote
    if runtime_profile == A100_PROBE_PROFILE:
        return train_stage1_a100_probe_remote, train_stage2_a100_probe_remote, train_stage3_a100_probe_remote
    if runtime_profile == A100_COMPAT_PROFILE:
        return train_stage1_a100_compat_remote, train_stage2_a100_compat_remote, train_stage3_a100_compat_remote
    if runtime_profile == H100_STABLE_PROFILE:
        return train_stage1_h100_stable_remote, train_stage2_h100_stable_remote, train_stage3_h100_stable_remote
    if runtime_profile == H100_PROBE_PROFILE:
        return train_stage1_h100_probe_remote, train_stage2_h100_probe_remote, train_stage3_h100_probe_remote
    return train_stage1_h100_compat_remote, train_stage2_h100_compat_remote, train_stage3_h100_compat_remote


def get_eval_sweep_remote_function(runtime_profile: str = DEFAULT_RUNTIME_PROFILE):
    get_runtime_profile(runtime_profile)
    if runtime_profile == T4_STABLE_PROFILE:
        return eval_sweep_t4_stable_remote
    if runtime_profile == A100_PROBE_PROFILE:
        return eval_sweep_a100_probe_remote
    if runtime_profile == A100_COMPAT_PROFILE:
        return eval_sweep_a100_compat_remote
    if runtime_profile == H100_STABLE_PROFILE:
        return eval_sweep_h100_stable_remote
    if runtime_profile == H100_PROBE_PROFILE:
        return eval_sweep_h100_probe_remote
    return eval_sweep_h100_compat_remote


def run_requested_stages(
    run_name: str,
    seed: int = 0,
    stage: str = "both",
    extra_args: tuple[str, ...] = (),
    runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
    tactile: bool = False,
    pointcloud_points: int = 1024,
    auto_eval: bool = False,
    auto_eval_num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    if stage not in ("1", "2", "3", "both", "all"):
        raise ValueError(f"Unsupported stage selection: {stage}")
    profile = get_runtime_profile(runtime_profile)
    stage1_remote, stage2_remote, stage3_remote = get_stage_remote_functions(profile.name)
    pointcloud_args = with_pointcloud_overrides(extra_args, pointcloud_points=pointcloud_points)
    eval_pointcloud_points = int(_override_value(pointcloud_args, "task.env.hora.nPointCloudPts") or pointcloud_points)

    if stage in ("1", "both", "all"):
        print(f"[hora] Starting stage 1 training: {run_name} [{profile.name}]")
        stage1_remote.remote(run_name, seed, pointcloud_args)

    if stage in ("2", "both", "all"):
        print(f"[hora] Starting stage 2 training: {run_name} [{profile.name}]")
        tactile_args = with_tactile_overrides(pointcloud_args, tactile=tactile)
        stage2_remote.remote(run_name, seed, tactile_args)

        if auto_eval:
            output_dir = get_auto_eval_output_dir(run_name)
            eval_remote = get_eval_sweep_remote_function(profile.name)
            print(f"[hora] Starting stage 2 BTG mean eval sweep: {run_name} [{profile.name}]")
            eval_remote.remote(
                "",
                output_dir,
                False,
                get_auto_eval_wandb_name(run_name),
                "eval",
                run_name,
                tactile_args,
                eval_pointcloud_points,
                auto_eval_num_seeds,
            )

    if stage in ("3", "all"):
        print(f"[hora] Starting stage 3 BTG13 fine-tuning: {run_name} [{profile.name}]")
        stage3_remote.remote(run_name, seed, with_tactile_overrides(pointcloud_args, tactile=tactile))

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
    image=RUNTIME_PROFILES[T4_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[T4_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[T4_STABLE_PROFILE].function_env,
))
def train_stage3_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 3: Fine-tune Stage 2 on BTG13."""
    emit_runtime_diagnostics(T4_STABLE_PROFILE)
    _run_stage(3, run_name, seed=seed, extra_args=extra_args)


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
    image=RUNTIME_PROFILES[A100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_PROBE_PROFILE].function_env,
))
def train_stage3_a100_probe_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 3 on the current image with an explicit A100 for diagnostics."""
    emit_runtime_diagnostics(A100_PROBE_PROFILE)
    _run_stage(3, run_name, seed=seed, extra_args=extra_args)


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


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[A100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_COMPAT_PROFILE].function_env,
))
def train_stage3_a100_compat_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 3 on the alternate A100 compatibility image."""
    emit_runtime_diagnostics(A100_COMPAT_PROFILE)
    _run_stage(3, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_STABLE_PROFILE].function_env,
))
def train_stage1_h100_stable_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1 on the validated H100 stable path."""
    emit_runtime_diagnostics(H100_STABLE_PROFILE)
    _run_stage(1, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_STABLE_PROFILE].function_env,
))
def train_stage2_h100_stable_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2 on the validated H100 stable path."""
    emit_runtime_diagnostics(H100_STABLE_PROFILE)
    _run_stage(2, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_STABLE_PROFILE].function_env,
))
def train_stage3_h100_stable_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 3 on the validated H100 stable path."""
    emit_runtime_diagnostics(H100_STABLE_PROFILE)
    _run_stage(3, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_PROBE_PROFILE].function_env,
))
def train_stage1_h100_probe_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1 on the current image with an explicit H100 for diagnostics."""
    emit_runtime_diagnostics(H100_PROBE_PROFILE)
    _run_stage(1, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_PROBE_PROFILE].function_env,
))
def train_stage2_h100_probe_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2 on the current image with an explicit H100 for diagnostics."""
    emit_runtime_diagnostics(H100_PROBE_PROFILE)
    _run_stage(2, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_PROBE_PROFILE].function_env,
))
def train_stage3_h100_probe_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 3 on the current image with an explicit H100 for diagnostics."""
    emit_runtime_diagnostics(H100_PROBE_PROFILE)
    _run_stage(3, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_COMPAT_PROFILE].function_env,
))
def train_stage1_h100_compat_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 1 on the alternate H100 compatibility image."""
    emit_runtime_diagnostics(H100_COMPAT_PROFILE)
    _run_stage(1, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_COMPAT_PROFILE].function_env,
))
def train_stage2_h100_compat_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 2 on the alternate H100 compatibility image."""
    emit_runtime_diagnostics(H100_COMPAT_PROFILE)
    _run_stage(2, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_COMPAT_PROFILE].function_env,
))
def train_stage3_h100_compat_remote(run_name: str, seed: int = 0, extra_args: tuple[str, ...] = ()):
    """Stage 3 on the alternate H100 compatibility image."""
    emit_runtime_diagnostics(H100_COMPAT_PROFILE)
    _run_stage(3, run_name, seed=seed, extra_args=extra_args)


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[T4_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[T4_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[T4_STABLE_PROFILE].function_env,
))
def eval_sweep_t4_stable_remote(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
    auto_run_name: str = "",
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    """Run a manifest-driven evaluation sweep on T4."""
    emit_runtime_diagnostics(T4_STABLE_PROFILE)
    _run_eval_sweep(
        manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        auto_run_name=auto_run_name,
        tactile_args=tactile_args,
        pointcloud_points=pointcloud_points,
        num_seeds=num_seeds,
    )


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[A100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_PROBE_PROFILE].function_env,
))
def eval_sweep_a100_probe_remote(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
    auto_run_name: str = "",
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    """Run a manifest-driven evaluation sweep on the current A100 image."""
    emit_runtime_diagnostics(A100_PROBE_PROFILE)
    _run_eval_sweep(
        manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        auto_run_name=auto_run_name,
        tactile_args=tactile_args,
        pointcloud_points=pointcloud_points,
        num_seeds=num_seeds,
    )


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[A100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[A100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[A100_COMPAT_PROFILE].function_env,
))
def eval_sweep_a100_compat_remote(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
    auto_run_name: str = "",
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    """Run a manifest-driven evaluation sweep on the alternate A100 compatibility image."""
    emit_runtime_diagnostics(A100_COMPAT_PROFILE)
    _run_eval_sweep(
        manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        auto_run_name=auto_run_name,
        tactile_args=tactile_args,
        pointcloud_points=pointcloud_points,
        num_seeds=num_seeds,
    )


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_STABLE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_STABLE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_STABLE_PROFILE].function_env,
))
def eval_sweep_h100_stable_remote(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
    auto_run_name: str = "",
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    """Run a manifest-driven evaluation sweep on the validated H100 image."""
    emit_runtime_diagnostics(H100_STABLE_PROFILE)
    _run_eval_sweep(
        manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        auto_run_name=auto_run_name,
        tactile_args=tactile_args,
        pointcloud_points=pointcloud_points,
        num_seeds=num_seeds,
    )


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_PROBE_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_PROBE_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_PROBE_PROFILE].function_env,
))
def eval_sweep_h100_probe_remote(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
    auto_run_name: str = "",
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    """Run a manifest-driven evaluation sweep on the current H100 image."""
    emit_runtime_diagnostics(H100_PROBE_PROFILE)
    _run_eval_sweep(
        manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        auto_run_name=auto_run_name,
        tactile_args=tactile_args,
        pointcloud_points=pointcloud_points,
        num_seeds=num_seeds,
    )


@app.function(**_modal_function_kwargs(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=RUNTIME_PROFILES[H100_COMPAT_PROFILE].image,
    secrets=function_secrets,
    gpu=RUNTIME_PROFILES[H100_COMPAT_PROFILE].gpu,
    function_env=RUNTIME_PROFILES[H100_COMPAT_PROFILE].function_env,
))
def eval_sweep_h100_compat_remote(
    manifest: str,
    output_dir: str = "",
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
    auto_run_name: str = "",
    tactile_args: tuple[str, ...] = (),
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    """Run a manifest-driven evaluation sweep on the alternate H100 compatibility image."""
    emit_runtime_diagnostics(H100_COMPAT_PROFILE)
    _run_eval_sweep(
        manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        auto_run_name=auto_run_name,
        tactile_args=tactile_args,
        pointcloud_points=pointcloud_points,
        num_seeds=num_seeds,
    )


@app.local_entrypoint()
def eval_sweep(
    manifest: str,
    output_dir: str = "",
    runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
):
    """
    Run a manifest-driven eval sweep on Modal.

    Args:
        manifest: Repo-relative manifest path, e.g. configs/eval_sweeps/btg13_tactile04201119_10seeds.json.
        output_dir: Optional output directory. Defaults to outputs/eval_sweeps/<manifest>_<timestamp>/ on the volume.
        runtime_profile: Modal runtime profile. One of t4_stable, a100_probe, a100_compat, h100_stable, h100_probe, h100_compat.
        dry_run: Prepare cases without running Isaac Gym evals.
        wandb_name: Optional W&B run name for summary tables and plots.
        wandb_group: W&B group for the eval summary run.
    """
    remote_fn = get_eval_sweep_remote_function(runtime_profile)
    remote_fn.remote(manifest, output_dir, dry_run, wandb_name, wandb_group)


@app.local_entrypoint()
def stage2_eval(
    run_name: str,
    overrides: str = "",
    runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
    tactile: bool = False,
    pointcloud_points: int = 1024,
    num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
    dry_run: bool = False,
):
    """
    Run only the automatic Stage 2 BTG mean-object eval sweep for an existing run.

    Args:
        run_name: Existing training run name with a Stage 2 checkpoint.
        overrides: Extra Hydra overrides used to infer tactile/point-cloud eval settings.
        runtime_profile: Modal runtime profile.
        tactile: When true, append Stage 2 tactile-history defaults before eval.
        pointcloud_points: Point cloud resolution for shape encoding. Must be 100 or 1024.
        num_seeds: Number of eval seeds per BTG object.
        dry_run: Prepare cases without running Isaac Gym evals.
    """
    extra_args = parse_overrides(overrides)
    pointcloud_args = with_pointcloud_overrides(extra_args, pointcloud_points=pointcloud_points)
    tactile_args = with_tactile_overrides(pointcloud_args, tactile=tactile)
    eval_pointcloud_points = int(_override_value(pointcloud_args, "task.env.hora.nPointCloudPts") or pointcloud_points)
    remote_fn = get_eval_sweep_remote_function(runtime_profile)
    remote_fn.remote(
        "",
        get_auto_eval_output_dir(run_name),
        dry_run,
        "" if dry_run else get_auto_eval_wandb_name(run_name),
        "eval",
        run_name,
        tactile_args,
        eval_pointcloud_points,
        num_seeds,
    )


@app.local_entrypoint()
def main(
    run_name: str,
    seed: int = 0,
    stage: str = "both",
    overrides: str = "",
    runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
    tactile: bool = False,
    pointcloud_points: int = 1024,
    auto_eval: bool = False,
    auto_eval_num_seeds: int = DEFAULT_AUTO_EVAL_NUM_SEEDS,
):
    """
    Train HORA on Modal.

    Args:
        run_name: Name for this training run (used in output paths and wandb).
        seed: Random seed (default: 0).
        stage: Which stage to train — "1", "2", "3", "both", or "all" (default: "both").
        overrides: Extra Hydra overrides passed to train.py.
        runtime_profile: Modal runtime profile. One of t4_stable, a100_probe, a100_compat, h100_stable, h100_probe, h100_compat.
        tactile: When true, append the split tactile obs/history flags to stages 2 and 3.
        pointcloud_points: Point cloud resolution for shape encoding. Must be 100 or 1024.
        auto_eval: Run a Stage 2 BTG1-BTG13 mean-object eval sweep after Stage 2 completes.
        auto_eval_num_seeds: Number of eval seeds per BTG object for --auto-eval.
    """
    run_requested_stages(
        run_name,
        seed=seed,
        stage=stage,
        extra_args=parse_overrides(overrides),
        runtime_profile=runtime_profile,
        tactile=tactile,
        pointcloud_points=pointcloud_points,
        auto_eval=auto_eval,
        auto_eval_num_seeds=auto_eval_num_seeds,
    )
