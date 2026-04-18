import json
from pathlib import Path

from hora.utils.eval_sweep import (
    build_case_name,
    build_eval_command,
    infer_output_name_from_checkpoint,
    load_manifest,
    parse_eval_metrics,
)


def test_load_manifest_applies_defaults(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "models": [{"name": "baseline", "checkpoint": "outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt"}],
                "objects": [{"name": "mean", "object_type": "custom_btg13_mean"}],
            }
        )
    )

    manifest = load_manifest(manifest_path)
    assert manifest["num_envs"] == 4096
    assert manifest["max_evaluate_envs"] == 20000
    assert manifest["seeds"] == [42]
    assert manifest["base_overrides"] == []


def test_infer_output_name_from_checkpoint_handles_stage2_paths():
    checkpoint = "outputs/AllegroHandHora/hora_v0.0.2/stage2_nn/model_best.ckpt"
    assert infer_output_name_from_checkpoint(checkpoint) == "AllegroHandHora/hora_v0.0.2"


def test_build_eval_command_includes_tactile_history_and_eval_budget():
    manifest = {
        "num_envs": 1024,
        "max_evaluate_envs": 5000,
        "base_overrides": ["task.env.reset_height_threshold=0.6"],
    }
    model = {
        "name": "tactile_stage2",
        "checkpoint": "outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt",
        "algo": "ProprioAdapt",
        "use_tactile": True,
    }
    obj = {"name": "mean", "object_type": "custom_btg13_mean"}

    command = build_eval_command(manifest, model, obj, seed=7, python_executable="/venv/bin/python")

    assert command[:2] == ["/venv/bin/python", "train.py"]
    assert "task.maxEvaluateEnvs=5000" in command
    assert "task.env.numEnvs=1024" in command
    assert "task.env.object.type=custom_btg13_mean" in command
    assert "task.env.hora.useTactile=True" in command
    assert "task.env.hora.useTactileObs=False" in command
    assert "task.env.hora.useTactileHist=True" in command
    assert "train.ppo.proprio_adapt=True" in command
    assert "train.ppo.output_name=AllegroHandHora/demo" in command
    assert "checkpoint=outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt" in command


def test_build_eval_command_supports_tactile_observation_release_mode():
    manifest = {
        "num_envs": 512,
        "max_evaluate_envs": 2000,
        "base_overrides": [],
    }
    model = {
        "name": "tactile_stage1",
        "checkpoint": "outputs/AllegroHandHora/double_tactile_s1_2/stage1_nn/best.pth",
        "algo": "PPO",
        "use_tactile_obs": True,
        "use_tactile_hist": False,
        "priv_info": True,
        "proprio_adapt": False,
    }
    obj = {"name": "mean", "object_type": "custom_btg13_mean"}

    command = build_eval_command(manifest, model, obj, seed=11, python_executable="/venv/bin/python")

    assert "task.env.hora.useTactile=False" in command
    assert "task.env.hora.useTactileObs=True" in command
    assert "task.env.hora.useTactileHist=False" in command
    assert "train.algo=PPO" in command
    assert "train.ppo.proprio_adapt=False" in command
    assert "checkpoint=outputs/AllegroHandHora/double_tactile_s1_2/stage1_nn/best.pth" in command


def test_build_case_name_is_stable():
    model = {"name": "baseline"}
    obj = {"name": "mean"}
    assert build_case_name(model, obj, 42) == "baseline__mean__seed42"


def test_parse_eval_metrics_uses_last_progress_line():
    output = """
progress 100 / 20000 | reward: 1.25 | eps length: 120.0 | rotate reward: 0.75 | lin vel (x100): 0.0500 | command torque: 0.42
progress 20000 / 20000 | reward: 2.50 | eps length: 240.0 | rotate reward: 1.50 | lin vel (x100): 0.0250 | command torque: 0.21
"""
    metrics = parse_eval_metrics(output)
    assert metrics == {
        "progress": 20000.0,
        "max_evaluate_envs": 20000.0,
        "reward": 2.5,
        "eps_length": 240.0,
        "rotate_reward": 1.5,
        "lin_vel_x100": 0.025,
        "command_torque": 0.21,
    }
