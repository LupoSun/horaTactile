#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

# Disable wandb side effects for local recording.
os.environ.setdefault("WANDB_MODE", "disabled")

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(REPO_ROOT / ".torch_extensions"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import isaacgym
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from isaacgym import gymapi
from omegaconf import OmegaConf
import torch

from hora.algo.padapt.padapt import ProprioAdapt
from hora.algo.ppo.ppo import PPO
from hora.tasks import isaacgym_task_map
from hora.utils.misc import set_np_formatting, set_seed
from hora.utils.reformat import omegaconf_to_dict
from hora.utils.recording import (
    build_recording_overrides,
    default_recording_path,
    resolve_checkpoint_and_output_name,
    rgba_image_to_rgb_array,
)


def _ensure_resolvers() -> None:
    if not OmegaConf.has_resolver("eq"):
        OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    if not OmegaConf.has_resolver("contains"):
        OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
    if not OmegaConf.has_resolver("if"):
        OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    if not OmegaConf.has_resolver("resolve_default"):
        OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)


def _build_camera(env, env_id: int, width: int, height: int, position: tuple[float, float, float], target: tuple[float, float, float]):
    cam_props = gymapi.CameraProperties()
    cam_props.width = width
    cam_props.height = height
    camera_handle = env.gym.create_camera_sensor(env.envs[env_id], cam_props)
    env.gym.set_camera_location(
        camera_handle,
        env.envs[env_id],
        gymapi.Vec3(*position),
        gymapi.Vec3(*target),
    )
    return camera_handle


def _capture_frame(env, env_id: int, camera_handle: int, width: int, height: int):
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    rgba = env.gym.get_camera_image(env.sim, env.envs[env_id], camera_handle, gymapi.IMAGE_COLOR)
    return rgba_image_to_rgb_array(rgba, width, height)


def _make_agent(env, output_dir: str, config):
    algo_cls = {"PPO": PPO, "ProprioAdapt": ProprioAdapt}[config.train.algo]
    agent = algo_cls(env, output_dir, full_config=config)
    agent.restore_test(config.train.load_path)
    agent.set_eval()
    return agent


def _policy_action(agent, obs_dict):
    if agent.__class__.__name__ == "PPO":
        input_dict = {
            "obs": agent.running_mean_std(obs_dict["obs"]),
            "priv_info": obs_dict["priv_info"],
        }
    else:
        input_dict = {
            "obs": agent.running_mean_std(obs_dict["obs"]),
            "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"].detach()),
        }
    mu = agent.model.act_inference(input_dict)
    return torch.clamp(mu, -1.0, 1.0)


def _open_writer(output_path: Path, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("MP4 output requires imageio-ffmpeg. Install it in the project venv.") from exc
    return imageio.get_writer(output_path, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Isaac Gym policy playback to frames or an animated output.")
    parser.add_argument("--run-name", help="Experiment run name under outputs/AllegroHandHora/<run_name>/")
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True, help="Training stage to record.")
    parser.add_argument("--checkpoint", help="Optional explicit checkpoint path. Overrides --run-name when provided.")
    parser.add_argument("--object-type", default="simple_tennis_ball", help="task.env.object.type override to render.")
    parser.add_argument("--tactile", action="store_true", help="Enable tactile history for Stage 2 playback.")
    parser.add_argument("--steps", type=int, default=400, help="Number of policy steps to record.")
    parser.add_argument("--frame-every", type=int, default=1, help="Capture every Nth policy step.")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS for GIF/MP4 output.")
    parser.add_argument("--width", type=int, default=960, help="Capture width in pixels.")
    parser.add_argument("--height", type=int, default=540, help="Capture height in pixels.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs. Recording uses env 0 by default.")
    parser.add_argument("--env-id", type=int, default=0, help="Environment index to record from.")
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        default=(0.0, 0.4, 1.5),
        metavar=("X", "Y", "Z"),
        help="Camera position in the selected env frame.",
    )
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.5),
        metavar=("X", "Y", "Z"),
        help="Camera target in the selected env frame.",
    )
    parser.add_argument("--output", type=Path, help="Animated output path (.gif or .mp4). Defaults to a GIF under outputs/recordings/.")
    parser.add_argument("--frames-dir", type=Path, help="Optional directory to save per-frame PNG files.")
    parser.add_argument("--overrides", default="", help='Extra Hydra overrides as a single shell-style string, e.g. \'task.env.reset_height_threshold=0.6\'.')
    args = parser.parse_args()

    if args.frame_every <= 0:
        raise SystemExit("--frame-every must be positive")
    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive")
    if not (0 <= args.env_id < args.num_envs):
        raise SystemExit("--env-id must be within [0, num_envs)")

    checkpoint, output_name = resolve_checkpoint_and_output_name(args.run_name, args.stage, checkpoint=args.checkpoint)
    output_path = args.output or default_recording_path(output_name, args.stage, args.object_type, ext="gif")
    extra_overrides = shlex.split(args.overrides)

    _ensure_resolvers()
    overrides = build_recording_overrides(
        output_name=output_name,
        checkpoint=checkpoint,
        stage=args.stage,
        object_type=args.object_type,
        use_tactile=args.tactile,
        num_envs=args.num_envs,
        extra_overrides=extra_overrides,
    )

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        config = compose(config_name="config", overrides=overrides)
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    set_np_formatting()
    config.seed = set_seed(config.seed)

    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )
    output_dir = os.path.join("outputs", config.train.ppo.output_name)
    agent = _make_agent(env, output_dir, config)
    camera_handle = _build_camera(env, args.env_id, args.width, args.height, tuple(args.camera_position), tuple(args.camera_target))

    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    writer = _open_writer(output_path, fps=args.fps)
    obs_dict = env.reset()
    frame_index = 0

    try:
        initial_frame = _capture_frame(env, args.env_id, camera_handle, args.width, args.height)
        writer.append_data(initial_frame)
        if args.frames_dir:
            imageio.imwrite(args.frames_dir / f"frame_{frame_index:05d}.png", initial_frame)
        frame_index += 1

        for step_idx in range(args.steps):
            mu = _policy_action(agent, obs_dict)
            obs_dict, _, _, _ = env.step(mu)
            if (step_idx + 1) % args.frame_every != 0:
                continue
            frame = _capture_frame(env, args.env_id, camera_handle, args.width, args.height)
            writer.append_data(frame)
            if args.frames_dir:
                imageio.imwrite(args.frames_dir / f"frame_{frame_index:05d}.png", frame)
            frame_index += 1
            print(f"Captured frame {frame_index} / step {step_idx + 1}")
    finally:
        writer.close()
        if getattr(env, "viewer", None):
            env.gym.destroy_viewer(env.viewer)

    print(f"Wrote recording: {output_path}")
    if args.frames_dir:
        print(f"Wrote frames to: {args.frames_dir}")


if __name__ == "__main__":
    main()
