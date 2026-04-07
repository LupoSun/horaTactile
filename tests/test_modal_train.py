from pathlib import Path
from types import SimpleNamespace

import pytest

import modal_train


def test_modal_train_module_exports_expected_entrypoints():
    assert modal_train.env["WANDB_DIR"] == f"{modal_train.VOLUME_PATH}/wandb"
    assert modal_train.DEFAULT_RUNTIME_PROFILE == modal_train.T4_STABLE_PROFILE
    assert hasattr(modal_train.train_stage1_remote, "remote")
    assert hasattr(modal_train.train_stage2_remote, "remote")
    assert hasattr(modal_train.train_stage1_a100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage2_a100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage1_a100_compat_remote, "remote")
    assert hasattr(modal_train.train_stage2_a100_compat_remote, "remote")


def test_runtime_profiles_are_explicit_and_validated():
    t4_profile = modal_train.get_runtime_profile(modal_train.T4_STABLE_PROFILE)
    a100_probe = modal_train.get_runtime_profile(modal_train.A100_PROBE_PROFILE)
    a100_compat = modal_train.get_runtime_profile(modal_train.A100_COMPAT_PROFILE)

    assert t4_profile.gpu == modal_train.T4_GPU
    assert a100_probe.gpu == modal_train.A100_PROBE_GPU
    assert a100_probe.function_env["CUDA_LAUNCH_BLOCKING"] == "1"
    assert a100_compat.gpu == modal_train.A100_COMPAT_GPU

    with pytest.raises(ValueError):
        modal_train.get_runtime_profile("bogus")


def test_parse_overrides_respects_shell_quoting():
    overrides = 'task.env.numEnvs=64 "train.notes=hello world"'
    assert modal_train.parse_overrides(overrides) == (
        "task.env.numEnvs=64",
        "train.notes=hello world",
    )


def test_expected_cache_files_match_default_config():
    assert modal_train.expected_cache_files() == (
        "internal_allegro_grasp_50k_s07.npy",
        "internal_allegro_grasp_50k_s072.npy",
        "internal_allegro_grasp_50k_s074.npy",
        "internal_allegro_grasp_50k_s076.npy",
        "internal_allegro_grasp_50k_s078.npy",
        "internal_allegro_grasp_50k_s08.npy",
        "internal_allegro_grasp_50k_s082.npy",
        "internal_allegro_grasp_50k_s084.npy",
        "internal_allegro_grasp_50k_s086.npy",
    )


def test_is_cache_complete_requires_full_default_set(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    expected_files = modal_train.expected_cache_files()

    (cache_dir / expected_files[0]).write_text("partial")
    assert not modal_train.is_cache_complete(str(cache_dir))

    for filename in expected_files[1:]:
        (cache_dir / filename).write_text("ready")
    assert modal_train.is_cache_complete(str(cache_dir))


def test_setup_project_symlinks_points_into_volume(tmp_path):
    project_dir = tmp_path / "project"
    volume_dir = tmp_path / "vol"
    project_dir.mkdir()

    modal_train.setup_project_symlinks(str(project_dir), str(volume_dir))

    outputs_link = project_dir / "outputs"
    cache_link = project_dir / "cache"
    assert outputs_link.is_symlink()
    assert cache_link.is_symlink()
    assert outputs_link.resolve() == volume_dir / "outputs"
    assert cache_link.resolve() == volume_dir / "cache"


def test_checkpoint_checks_use_stage_specific_best_files(tmp_path):
    volume_dir = tmp_path / "vol"
    stage1_best = Path(modal_train.get_stage_best_checkpoint_volume_path("exp", 1, str(volume_dir)))
    stage2_best = Path(modal_train.get_stage_best_checkpoint_volume_path("exp", 2, str(volume_dir)))
    stage1_best.parent.mkdir(parents=True, exist_ok=True)
    stage2_best.parent.mkdir(parents=True, exist_ok=True)

    modal_train.check_no_overwrite("exp", 1, str(volume_dir))
    modal_train.check_no_overwrite("exp", 2, str(volume_dir))

    (stage2_best.parent / "best.pth").write_text("wrong-stage2-name")
    modal_train.check_no_overwrite("exp", 2, str(volume_dir))

    stage1_best.write_text("stage1")
    with pytest.raises(RuntimeError):
        modal_train.check_no_overwrite("exp", 1, str(volume_dir))

    stage2_best.write_text("stage2")
    with pytest.raises(RuntimeError):
        modal_train.check_no_overwrite("exp", 2, str(volume_dir))


def test_check_stage1_exists_requires_best_pth(tmp_path):
    volume_dir = tmp_path / "vol"
    with pytest.raises(RuntimeError):
        modal_train.check_stage1_exists("demo", str(volume_dir))

    stage1_best = Path(modal_train.get_stage_best_checkpoint_volume_path("demo", 1, str(volume_dir)))
    stage1_best.parent.mkdir(parents=True, exist_ok=True)
    stage1_best.write_text("ready")
    modal_train.check_stage1_exists("demo", str(volume_dir))


def test_build_stage_commands_include_journal_defaults():
    stage1_cmd = modal_train.build_stage1_command("demo", seed=7, extra_args=("task.env.numEnvs=64",))
    stage2_cmd = modal_train.build_stage2_command("demo", seed=11, extra_args=("train.ppo.max_agent_steps=1024",))

    assert stage1_cmd[:2] == [modal_train.CONDA_PYTHON, "train.py"]
    assert "task=AllegroHandHora" in stage1_cmd
    assert "headless=True" in stage1_cmd
    assert "train.algo=PPO" in stage1_cmd
    assert "train.ppo.priv_info=True" in stage1_cmd
    assert "train.ppo.proprio_adapt=False" in stage1_cmd
    assert "task.env.object.type=cylinder_default" in stage1_cmd
    assert "train.ppo.output_name=AllegroHandHora/demo" in stage1_cmd
    assert stage1_cmd[-1] == "task.env.numEnvs=64"

    assert "train.algo=ProprioAdapt" in stage2_cmd
    assert "task.env.numEnvs=20000" in stage2_cmd
    assert "train.ppo.proprio_adapt=True" in stage2_cmd
    assert "checkpoint=outputs/AllegroHandHora/demo/stage1_nn/best.pth" in stage2_cmd
    assert stage2_cmd[-1] == "train.ppo.max_agent_steps=1024"


def test_run_requested_stages_dispatches_requested_remote_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage2", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=3,
        stage="both",
        extra_args=("task.env.numEnvs=64",),
        runtime_profile=modal_train.T4_STABLE_PROFILE,
    )

    assert calls == [
        ("stage1", "demo", 3, ("task.env.numEnvs=64",)),
        ("stage2", "demo", 3, ("task.env.numEnvs=64",)),
    ]


def test_run_requested_stages_uses_selected_a100_profile(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_a100_probe_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("probe-stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_a100_probe_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("probe-stage2", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=9,
        stage="both",
        extra_args=("train.ppo.max_agent_steps=1024",),
        runtime_profile=modal_train.A100_PROBE_PROFILE,
    )

    assert calls == [
        ("probe-stage1", "demo", 9, ("train.ppo.max_agent_steps=1024",)),
        ("probe-stage2", "demo", 9, ("train.ppo.max_agent_steps=1024",)),
    ]


def test_main_parses_overrides_before_dispatch(monkeypatch):
    captured = {}

    def fake_run_requested_stages(run_name, seed=0, stage="both", extra_args=(), runtime_profile=modal_train.DEFAULT_RUNTIME_PROFILE):
        captured["run_name"] = run_name
        captured["seed"] = seed
        captured["stage"] = stage
        captured["extra_args"] = extra_args
        captured["runtime_profile"] = runtime_profile

    monkeypatch.setattr(modal_train, "run_requested_stages", fake_run_requested_stages)

    modal_train.main(
        run_name="demo",
        seed=5,
        stage="2",
        overrides='task.env.numEnvs=64 "train.notes=hello world"',
        runtime_profile=modal_train.A100_COMPAT_PROFILE,
    )

    assert captured == {
        "run_name": "demo",
        "seed": 5,
        "stage": "2",
        "extra_args": ("task.env.numEnvs=64", "train.notes=hello world"),
        "runtime_profile": modal_train.A100_COMPAT_PROFILE,
    }
