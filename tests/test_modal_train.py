from pathlib import Path
from types import SimpleNamespace

import pytest

import modal_train

DEFAULT_POINTCLOUD_ARGS = (
    "task.env.hora.nPointCloudPts=1024",
    "train.ppo.n_pointcloud_pts=1024",
)


def test_modal_train_module_exports_expected_entrypoints():
    assert modal_train.env["WANDB_DIR"] == f"{modal_train.VOLUME_PATH}/wandb"
    assert modal_train.DEFAULT_RUNTIME_PROFILE == modal_train.T4_STABLE_PROFILE
    assert hasattr(modal_train.train_stage1_remote, "remote")
    assert hasattr(modal_train.train_stage2_remote, "remote")
    assert hasattr(modal_train.train_stage3_remote, "remote")
    assert hasattr(modal_train.train_stage1_a100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage2_a100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage3_a100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage1_a100_compat_remote, "remote")
    assert hasattr(modal_train.train_stage2_a100_compat_remote, "remote")
    assert hasattr(modal_train.train_stage3_a100_compat_remote, "remote")
    assert hasattr(modal_train.train_stage1_h100_stable_remote, "remote")
    assert hasattr(modal_train.train_stage2_h100_stable_remote, "remote")
    assert hasattr(modal_train.train_stage3_h100_stable_remote, "remote")
    assert hasattr(modal_train.train_stage1_h100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage2_h100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage3_h100_probe_remote, "remote")
    assert hasattr(modal_train.train_stage1_h100_compat_remote, "remote")
    assert hasattr(modal_train.train_stage2_h100_compat_remote, "remote")
    assert hasattr(modal_train.train_stage3_h100_compat_remote, "remote")


def test_runtime_profiles_are_explicit_and_validated():
    t4_profile = modal_train.get_runtime_profile(modal_train.T4_STABLE_PROFILE)
    a100_probe = modal_train.get_runtime_profile(modal_train.A100_PROBE_PROFILE)
    a100_compat = modal_train.get_runtime_profile(modal_train.A100_COMPAT_PROFILE)
    h100_stable = modal_train.get_runtime_profile(modal_train.H100_STABLE_PROFILE)
    h100_probe = modal_train.get_runtime_profile(modal_train.H100_PROBE_PROFILE)
    h100_compat = modal_train.get_runtime_profile(modal_train.H100_COMPAT_PROFILE)

    assert t4_profile.gpu == modal_train.T4_GPU
    assert a100_probe.gpu == modal_train.A100_PROBE_GPU
    assert a100_probe.function_env["CUDA_LAUNCH_BLOCKING"] == "1"
    assert a100_compat.gpu == modal_train.A100_COMPAT_GPU
    assert h100_stable.gpu == modal_train.H100_STABLE_GPU
    assert "CUDA_LAUNCH_BLOCKING" not in h100_stable.function_env
    assert h100_probe.gpu == modal_train.H100_PROBE_GPU
    assert h100_probe.function_env["CUDA_LAUNCH_BLOCKING"] == "1"
    assert h100_compat.gpu == modal_train.H100_COMPAT_GPU

    with pytest.raises(ValueError):
        modal_train.get_runtime_profile("bogus")


def test_parse_overrides_respects_shell_quoting():
    overrides = 'task.env.numEnvs=64 "train.notes=hello world"'
    assert modal_train.parse_overrides(overrides) == (
        "task.env.numEnvs=64",
        "train.notes=hello world",
    )


def test_parse_modal_cpu_env_value():
    assert modal_train._parse_modal_cpu(None) is None
    assert modal_train._parse_modal_cpu("") is None
    assert modal_train._parse_modal_cpu("8") == 8.0
    assert modal_train._parse_modal_cpu("8.5") == 8.5

    with pytest.raises(ValueError):
        modal_train._parse_modal_cpu("0")


def test_with_tactile_overrides_appends_tactile_once():
    assert modal_train.with_tactile_overrides(("train.ppo.max_agent_steps=1024",), tactile=False) == (
        "train.ppo.max_agent_steps=1024",
    )
    assert modal_train.with_tactile_overrides(("train.ppo.max_agent_steps=1024",), tactile=True) == (
        "train.ppo.max_agent_steps=1024",
        "task.env.hora.useTactileObs=True",
        "task.env.hora.useTactileHist=True",
    )
    assert modal_train.with_tactile_overrides(("task.env.hora.useTactileObs=False",), tactile=True) == (
        "task.env.hora.useTactileObs=False",
        "task.env.hora.useTactileHist=True",
    )
    assert modal_train.with_tactile_overrides((), tactile=True, tactile_hist=False) == (
        "task.env.hora.useTactileObs=True",
        "task.env.hora.useTactileHist=False",
    )


def test_with_pointcloud_overrides_selects_supported_resolution():
    assert modal_train.with_pointcloud_overrides(()) == (
        "task.env.hora.nPointCloudPts=1024",
        "train.ppo.n_pointcloud_pts=1024",
    )
    assert modal_train.with_pointcloud_overrides((), pointcloud_points=100) == (
        "task.env.hora.nPointCloudPts=100",
        "train.ppo.n_pointcloud_pts=100",
    )
    assert modal_train.with_pointcloud_overrides((), pointcloud_points=200) == (
        "task.env.hora.nPointCloudPts=200",
        "train.ppo.n_pointcloud_pts=200",
    )
    assert modal_train.with_pointcloud_overrides((), pointcloud_points=300) == (
        "task.env.hora.nPointCloudPts=300",
        "train.ppo.n_pointcloud_pts=300",
    )
    assert modal_train.with_pointcloud_overrides((), pointcloud_points=500) == (
        "task.env.hora.nPointCloudPts=500",
        "train.ppo.n_pointcloud_pts=500",
    )
    assert modal_train.with_pointcloud_overrides(("task.env.hora.nPointCloudPts=1024",), pointcloud_points=100) == (
        "task.env.hora.nPointCloudPts=1024",
        "train.ppo.n_pointcloud_pts=100",
    )
    with pytest.raises(ValueError):
        modal_train.with_pointcloud_overrides((), pointcloud_points=512)


def test_with_rl_variant_overrides_selects_supported_presets():
    assert modal_train.with_rl_variant_overrides(("train.ppo.max_agent_steps=1024",)) == (
        "train.ppo.max_agent_steps=1024",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_RECURRENT) == (
        "train.ppo.recurrent_obs=True",
        "train.ppo.recurrent_obs_seq_len=3",
        "train.ppo.recurrent_hidden_size=128",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_ASYM_CRITIC) == (
        "train.ppo.asymmetric_critic=True",
        "train.ppo.actor_use_privileged_info=False",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_TUNED) == (
        "train.ppo.kl_threshold=0.01",
        "train.ppo.entropy_coef=0.001",
        "train.ppo.horizon_length=16",
        "train.ppo.minibatch_size=32768",
        "train.ppo.mini_epochs=4",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_CONTACT_GATED) == (
        "train.ppo.contact_event_gating=True",
        "train.ppo.contact_num_modes=4",
        "train.ppo.contact_gate_hidden_size=32",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_history_len=3",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_ASYM_CONTACT_GATED_V2) == (
        "train.ppo.asymmetric_critic=True",
        "train.ppo.actor_use_privileged_info=False",
        "train.ppo.contact_event_gating=True",
        "train.ppo.contact_num_modes=4",
        "train.ppo.contact_gate_hidden_size=32",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_history_len=3",
        "train.ppo.contact_gate_event_features=True",
        "train.ppo.contact_gate_threshold=0.05",
        "train.ppo.contact_gate_balance_coef=0.01",
        "train.ppo.contact_gate_switch_coef=0.005",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_CONTACT_OPTIONS) == (
        "train.ppo.contact_options=True",
        "train.ppo.contact_num_modes=4",
        "train.ppo.contact_gate_hidden_size=32",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_history_len=3",
        "train.ppo.contact_gate_event_features=True",
        "train.ppo.contact_gate_threshold=0.05",
        "train.ppo.contact_option_max_dwell=12",
        "train.ppo.contact_option_min_dwell=2",
        "train.ppo.contact_option_boundary_mode=soft",
        "train.ppo.contact_option_entropy_coef=0.002",
        "train.ppo.contact_termination_entropy_coef=0.001",
        "train.ppo.contact_termination_sparsity_coef=0.01",
        "train.ppo.contact_min_dwell_loss_coef=0.0",
        "train.ppo.contact_option_balance_coef=0.0",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_ASYM_CONTACT_OPTIONS) == (
        "train.ppo.asymmetric_critic=True",
        "train.ppo.actor_use_privileged_info=False",
        "train.ppo.contact_options=True",
        "train.ppo.contact_num_modes=4",
        "train.ppo.contact_gate_hidden_size=32",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_history_len=3",
        "train.ppo.contact_gate_event_features=True",
        "train.ppo.contact_gate_threshold=0.05",
        "train.ppo.contact_option_max_dwell=12",
        "train.ppo.contact_option_min_dwell=2",
        "train.ppo.contact_option_boundary_mode=soft",
        "train.ppo.contact_option_entropy_coef=0.002",
        "train.ppo.contact_termination_entropy_coef=0.001",
        "train.ppo.contact_termination_sparsity_coef=0.01",
        "train.ppo.contact_min_dwell_loss_coef=0.0",
        "train.ppo.contact_option_balance_coef=0.0",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_CONTACT_OPTIONS_V2) == (
        "train.ppo.contact_options=True",
        "train.ppo.contact_num_modes=4",
        "train.ppo.contact_gate_hidden_size=32",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_history_len=3",
        "train.ppo.contact_gate_event_features=True",
        "train.ppo.contact_gate_threshold=0.05",
        "train.ppo.contact_option_max_dwell=8",
        "train.ppo.contact_option_min_dwell=3",
        "train.ppo.contact_option_boundary_mode=forced",
        "train.ppo.contact_option_entropy_coef=0.004",
        "train.ppo.contact_termination_entropy_coef=0.001",
        "train.ppo.contact_termination_sparsity_coef=0.02",
        "train.ppo.contact_min_dwell_loss_coef=0.01",
        "train.ppo.contact_option_balance_coef=0.02",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_ASYM_CONTACT_OPTIONS_V2) == (
        "train.ppo.asymmetric_critic=True",
        "train.ppo.actor_use_privileged_info=False",
        "train.ppo.contact_options=True",
        "train.ppo.contact_num_modes=4",
        "train.ppo.contact_gate_hidden_size=32",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_history_len=3",
        "train.ppo.contact_gate_event_features=True",
        "train.ppo.contact_gate_threshold=0.05",
        "train.ppo.contact_option_max_dwell=8",
        "train.ppo.contact_option_min_dwell=3",
        "train.ppo.contact_option_boundary_mode=forced",
        "train.ppo.contact_option_entropy_coef=0.004",
        "train.ppo.contact_termination_entropy_coef=0.001",
        "train.ppo.contact_termination_sparsity_coef=0.02",
        "train.ppo.contact_min_dwell_loss_coef=0.01",
        "train.ppo.contact_option_balance_coef=0.02",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_CONTACT_RESET) == (
        "train.ppo.recurrent_obs=True",
        "train.ppo.recurrent_obs_seq_len=3",
        "train.ppo.recurrent_hidden_size=128",
        "train.ppo.contact_reset_recurrent=True",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_gate_hidden_size=32",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_PPO_CONTACT_AUX) == (
        "train.ppo.contact_transition_aux_loss=True",
        "train.ppo.contact_transition_aux_coef=0.05",
        "train.ppo.contact_transition_aux_threshold=0.05",
        "train.ppo.contact_tactile_dim=12",
        "train.ppo.contact_history_len=3",
    )
    assert modal_train.with_rl_variant_overrides((), rl_variant=modal_train.RL_VARIANT_TD3) == (
        "train.algo=TD3",
        "train.ppo.td3_batch_size=32768",
        "train.ppo.td3_learning_starts=80000",
        "train.ppo.td3_replay_size=100000",
    )
    assert modal_train.with_rl_variant_overrides(
        ("train.ppo.entropy_coef=0.01",),
        rl_variant=modal_train.RL_VARIANT_PPO_TUNED,
    )[1:] == (
        "train.ppo.kl_threshold=0.01",
        "train.ppo.horizon_length=16",
        "train.ppo.minibatch_size=32768",
        "train.ppo.mini_epochs=4",
    )
    with pytest.raises(ValueError):
        modal_train.with_rl_variant_overrides((), rl_variant="sac")


def test_build_auto_eval_manifest_targets_btg_mean_stage2():
    tactile_args = (
        "task.env.hora.useTactileObs=True",
        "task.env.hora.useTactileHist=True",
        "train.ppo.asymmetric_critic=True",
        "train.ppo.actor_use_privileged_info=False",
    )
    manifest = modal_train.build_auto_eval_manifest(
        "demo",
        tactile_args=tactile_args,
        pointcloud_points=100,
        num_seeds=3,
    )

    assert manifest["seeds"] == [0, 1, 2]
    assert len(manifest["objects"]) == 13
    assert manifest["objects"][0] == {
        "name": "btg1_mean",
        "object_type": "custom_btg1_mean",
        "object_index": 1,
    }
    assert manifest["objects"][-1]["object_type"] == "custom_btg13_mean"
    assert manifest["models"][0]["checkpoint"] == "outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt"
    assert manifest["models"][0]["use_tactile_obs"] is True
    assert manifest["models"][0]["use_tactile_hist"] is True
    assert manifest["models"][0]["use_shape_priv_info"] is True
    assert manifest["models"][0]["env_use_shape_priv_info"] is False
    assert manifest["models"][0]["extra_overrides"] == [
        "train.ppo.asymmetric_critic=True",
        "train.ppo.actor_use_privileged_info=False",
        "task.env.hora.nPointCloudPts=100",
        "train.ppo.n_pointcloud_pts=100",
    ]
    with pytest.raises(ValueError):
        modal_train.build_auto_eval_manifest("demo", num_seeds=0)


def test_build_auto_eval_manifest_preserves_recurrent_variant_overrides():
    manifest = modal_train.build_auto_eval_manifest(
        "demo",
        tactile_args=(
            "task.env.hora.useTactileObs=True",
            "task.env.hora.useTactileHist=True",
            "train.ppo.recurrent_obs=True",
            "train.ppo.recurrent_obs_seq_len=3",
            "train.ppo.recurrent_hidden_size=128",
            "task.env.hora.nPointCloudPts=500",
            "train.ppo.n_pointcloud_pts=500",
        ),
        pointcloud_points=500,
    )

    assert manifest["models"][0]["extra_overrides"] == [
        "train.ppo.recurrent_obs=True",
        "train.ppo.recurrent_obs_seq_len=3",
        "train.ppo.recurrent_hidden_size=128",
        "task.env.hora.nPointCloudPts=500",
        "train.ppo.n_pointcloud_pts=500",
    ]


def test_eval_pointcloud_preflight_ignores_builtin_ball(monkeypatch, tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "models": [
    {
      "use_shape_priv_info": true,
      "extra_overrides": ["task.env.hora.nPointCloudPts=100"]
    }
  ],
  "objects": [
    {"name": "btg1_mean", "object_type": "custom_btg1_mean"}
  ]
}
"""
    )
    loaded_assets = []

    def fake_catalog(object_type, sample_prob, repo_root):
        assert object_type == "custom_btg1_mean"
        return (
            ["custom_btg1_mean_0"],
            [1.0],
            {
                "simple_tennis_ball": "assets/ball.urdf",
                "custom_btg1_mean_0": "assets/custom/btg1_mean/BTG_1/BTG_1.urdf",
            },
        )

    def fake_load(asset_file, n_points, repo_root):
        loaded_assets.append(asset_file)

    monkeypatch.setattr(modal_train, "PROJECT_DIR", str(project_dir))
    monkeypatch.setattr(
        "hora.utils.object_assets.build_object_asset_catalog",
        fake_catalog,
    )
    monkeypatch.setattr(
        "hora.utils.object_assets.load_object_point_cloud",
        fake_load,
    )

    modal_train._ensure_eval_pointcloud_sidecars(str(manifest_path))

    assert loaded_assets == ["assets/custom/btg1_mean/BTG_1/BTG_1.urdf"]


def test_eval_pointcloud_preflight_skips_stage2_sensor_eval(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "models": [
    {
      "use_shape_priv_info": true,
      "env_use_shape_priv_info": false
    }
  ],
  "objects": [
    {"name": "btg1_mean", "object_type": "custom_btg1_mean"}
  ]
}
"""
    )

    def fail_catalog(*args, **kwargs):
        raise AssertionError("point-cloud preflight should not inspect assets")

    monkeypatch.setattr(
        "hora.utils.object_assets.build_object_asset_catalog",
        fail_catalog,
    )

    modal_train._ensure_eval_pointcloud_sidecars(str(manifest_path))


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
    stage3_best = Path(modal_train.get_stage_best_checkpoint_volume_path("exp", 3, str(volume_dir)))
    stage1_best.parent.mkdir(parents=True, exist_ok=True)
    stage2_best.parent.mkdir(parents=True, exist_ok=True)
    stage3_best.parent.mkdir(parents=True, exist_ok=True)

    modal_train.check_no_overwrite("exp", 1, str(volume_dir))
    modal_train.check_no_overwrite("exp", 2, str(volume_dir))
    modal_train.check_no_overwrite("exp", 3, str(volume_dir))

    (stage2_best.parent / "best.pth").write_text("wrong-stage2-name")
    modal_train.check_no_overwrite("exp", 2, str(volume_dir))

    stage1_best.write_text("stage1")
    with pytest.raises(RuntimeError):
        modal_train.check_no_overwrite("exp", 1, str(volume_dir))

    stage2_best.write_text("stage2")
    with pytest.raises(RuntimeError):
        modal_train.check_no_overwrite("exp", 2, str(volume_dir))

    stage3_best.write_text("stage3")
    with pytest.raises(RuntimeError):
        modal_train.check_no_overwrite("exp", 3, str(volume_dir))


def test_check_stage1_exists_requires_best_pth(tmp_path):
    volume_dir = tmp_path / "vol"
    with pytest.raises(RuntimeError):
        modal_train.check_stage1_exists("demo", str(volume_dir))

    stage1_best = Path(modal_train.get_stage_best_checkpoint_volume_path("demo", 1, str(volume_dir)))
    stage1_best.parent.mkdir(parents=True, exist_ok=True)
    stage1_best.write_text("ready")
    modal_train.check_stage1_exists("demo", str(volume_dir))


def test_check_stage2_exists_requires_model_best(tmp_path):
    volume_dir = tmp_path / "vol"
    with pytest.raises(RuntimeError):
        modal_train.check_stage2_exists("demo", str(volume_dir))

    stage2_best = Path(modal_train.get_stage_best_checkpoint_volume_path("demo", 2, str(volume_dir)))
    stage2_best.parent.mkdir(parents=True, exist_ok=True)
    stage2_best.write_text("ready")
    modal_train.check_stage2_exists("demo", str(volume_dir))


def test_build_stage_commands_include_journal_defaults():
    stage1_cmd = modal_train.build_stage1_command("demo", seed=7, extra_args=("task.env.numEnvs=64",))
    stage2_cmd = modal_train.build_stage2_command("demo", seed=11, extra_args=("train.ppo.max_agent_steps=1024",))
    stage3_cmd = modal_train.build_stage3_command("demo", seed=13, extra_args=("train.ppo.max_agent_steps=2048",))

    assert stage1_cmd[:2] == [modal_train.CONDA_PYTHON, "train.py"]
    assert "task=AllegroHandHora" in stage1_cmd
    assert "headless=True" in stage1_cmd
    assert "train.algo=PPO" in stage1_cmd
    assert "train.ppo.priv_info=True" in stage1_cmd
    assert "train.ppo.proprio_adapt=False" in stage1_cmd
    assert f"task.env.object.type={modal_train.TACTILE_CYLINDER_OBJECT_TYPE}" in stage1_cmd
    assert f"task.env.object.sampleProb={modal_train.TACTILE_CYLINDER_SAMPLE_PROB}" in stage1_cmd
    assert "task.env.hora.useShapePrivInfo=True" in stage1_cmd
    assert "task.env.hora.useExtendedPrivInfo=True" in stage1_cmd
    assert "task.env.hora.privInfoDim=17" in stage1_cmd
    assert "train.ppo.use_shape_priv_info=True" in stage1_cmd
    assert "train.ppo.priv_info_dim=17" in stage1_cmd
    assert "train.ppo.output_name=AllegroHandHora/demo" in stage1_cmd
    assert stage1_cmd[-1] == "task.env.numEnvs=64"

    assert "train.algo=ProprioAdapt" in stage2_cmd
    assert "task.env.numEnvs=20000" in stage2_cmd
    assert f"task.env.object.type={modal_train.TACTILE_CYLINDER_OBJECT_TYPE}" in stage2_cmd
    assert "task.env.hora.useShapePrivInfo=True" in stage2_cmd
    assert "task.env.hora.useExtendedPrivInfo=True" in stage2_cmd
    assert "train.ppo.use_shape_priv_info=True" in stage2_cmd
    assert "train.ppo.priv_info_dim=17" in stage2_cmd
    assert "train.ppo.proprio_adapt=True" in stage2_cmd
    assert "checkpoint=outputs/AllegroHandHora/demo/stage1_nn/best.pth" in stage2_cmd
    assert stage2_cmd[-1] == "train.ppo.max_agent_steps=1024"

    assert "train.algo=ProprioAdapt" in stage3_cmd
    assert "task.env.numEnvs=4096" in stage3_cmd
    assert "task.env.object.type=custom_btg13_mean" in stage3_cmd
    assert "task.env.forceScale=0.0" in stage3_cmd
    assert "task.env.randomization.jointNoiseScale=0.0" in stage3_cmd
    assert "train.ppo.nn_dir=stage3_nn" in stage3_cmd
    assert "train.ppo.wandb_group=stage3" in stage3_cmd
    assert "checkpoint=outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt" in stage3_cmd
    assert stage3_cmd[-1] == "train.ppo.max_agent_steps=2048"


def test_build_stage2_command_can_enable_tactile():
    stage2_cmd = modal_train.build_stage2_command("demo", tactile=True)
    assert "task.env.hora.useTactileObs=True" in stage2_cmd
    assert "task.env.hora.useTactileHist=True" in stage2_cmd


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
    monkeypatch.setattr(
        modal_train,
        "train_stage3_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage3", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=3,
        stage="both",
        extra_args=("task.env.numEnvs=64",),
        runtime_profile=modal_train.T4_STABLE_PROFILE,
    )

    assert calls == [
        ("stage1", "demo", 3, ("task.env.numEnvs=64", *DEFAULT_POINTCLOUD_ARGS)),
        ("stage2", "demo", 3, ("task.env.numEnvs=64", *DEFAULT_POINTCLOUD_ARGS)),
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
    monkeypatch.setattr(
        modal_train,
        "train_stage3_a100_probe_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("probe-stage3", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=9,
        stage="both",
        extra_args=("train.ppo.max_agent_steps=1024",),
        runtime_profile=modal_train.A100_PROBE_PROFILE,
    )

    assert calls == [
        ("probe-stage1", "demo", 9, ("train.ppo.max_agent_steps=1024", *DEFAULT_POINTCLOUD_ARGS)),
        ("probe-stage2", "demo", 9, ("train.ppo.max_agent_steps=1024", *DEFAULT_POINTCLOUD_ARGS)),
    ]


def test_run_requested_stages_uses_selected_h100_profile(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("h100-stable-stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("h100-stable-stage2", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage3_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("h100-stable-stage3", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=4,
        stage="both",
        extra_args=("train.ppo.max_agent_steps=1024",),
        runtime_profile=modal_train.H100_STABLE_PROFILE,
    )

    assert calls == [
        ("h100-stable-stage1", "demo", 4, ("train.ppo.max_agent_steps=1024", *DEFAULT_POINTCLOUD_ARGS)),
        ("h100-stable-stage2", "demo", 4, ("train.ppo.max_agent_steps=1024", *DEFAULT_POINTCLOUD_ARGS)),
    ]


def test_run_requested_stages_applies_rl_variant_to_stage1_and_stage2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage2", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage3_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage3", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=6,
        stage="both",
        runtime_profile=modal_train.H100_STABLE_PROFILE,
        rl_variant=modal_train.RL_VARIANT_PPO_RECURRENT,
    )

    variant_args = (
        "train.ppo.recurrent_obs=True",
        "train.ppo.recurrent_obs_seq_len=3",
        "train.ppo.recurrent_hidden_size=128",
    )
    assert calls == [
        ("stage1", "demo", 6, (*variant_args, *DEFAULT_POINTCLOUD_ARGS)),
        ("stage2", "demo", 6, (*variant_args, *DEFAULT_POINTCLOUD_ARGS)),
    ]


def test_run_requested_stages_keeps_td3_stage1_only_overrides_out_of_stage2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage2", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage3_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage3", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=6,
        stage="both",
        runtime_profile=modal_train.H100_STABLE_PROFILE,
        rl_variant=modal_train.RL_VARIANT_TD3,
    )

    assert calls == [
        (
            "stage1",
            "demo",
            6,
            (
                "train.algo=TD3",
                "train.ppo.td3_batch_size=32768",
                "train.ppo.td3_learning_starts=80000",
                "train.ppo.td3_replay_size=100000",
                *DEFAULT_POINTCLOUD_ARGS,
            ),
        ),
        ("stage2", "demo", 6, DEFAULT_POINTCLOUD_ARGS),
    ]


def test_run_requested_stages_applies_tactile_to_stage1_and_stage2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage2", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage3_h100_stable_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage3", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=6,
        stage="both",
        extra_args=("train.ppo.max_agent_steps=1024",),
        runtime_profile=modal_train.H100_STABLE_PROFILE,
        tactile=True,
    )

    assert calls == [
        (
            "stage1",
            "demo",
            6,
            (
                "train.ppo.max_agent_steps=1024",
                *DEFAULT_POINTCLOUD_ARGS,
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=False",
            ),
        ),
        (
            "stage2",
            "demo",
            6,
            (
                "train.ppo.max_agent_steps=1024",
                *DEFAULT_POINTCLOUD_ARGS,
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=True",
            ),
        ),
    ]


def test_run_requested_stages_can_auto_eval_after_stage2(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_a100_compat_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_a100_compat_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage2", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage3_a100_compat_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage3", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "eval_sweep_a100_compat_remote",
        SimpleNamespace(
            remote=lambda manifest, output_dir, dry_run, wandb_name, wandb_group, auto_run_name, tactile_args, pointcloud_points, num_seeds: calls.append(
                ("eval", manifest, output_dir, dry_run, wandb_name, wandb_group, auto_run_name, tactile_args, pointcloud_points, num_seeds)
            )
        ),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=6,
        stage="both",
        runtime_profile=modal_train.A100_COMPAT_PROFILE,
        tactile=True,
        pointcloud_points=100,
        auto_eval=True,
        auto_eval_num_seeds=7,
    )

    assert calls == [
        (
            "stage1",
            "demo",
            6,
            (
                "task.env.hora.nPointCloudPts=100",
                "train.ppo.n_pointcloud_pts=100",
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=False",
            ),
        ),
        (
            "stage2",
            "demo",
            6,
            (
                "task.env.hora.nPointCloudPts=100",
                "train.ppo.n_pointcloud_pts=100",
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=True",
            ),
        ),
        (
            "eval",
            "",
            f"{modal_train.VOLUME_PATH}/outputs/AllegroHandHora/demo/stage2_eval",
            False,
            "AllegroHandHora/demo_eval",
            "eval",
            "demo",
            (
                "task.env.hora.nPointCloudPts=100",
                "train.ppo.n_pointcloud_pts=100",
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=True",
            ),
            100,
            7,
        ),
    ]


def test_run_requested_stages_durable_uses_cloud_pipeline_for_multi_stage(monkeypatch):
    calls = []
    fake_call = SimpleNamespace(object_id="fc-123")
    monkeypatch.setattr(
        modal_train,
        "train_pipeline_a100_compat_remote",
        SimpleNamespace(spawn=lambda *args: calls.append(args) or fake_call),
    )

    result = modal_train.run_requested_stages_durable(
        "demo",
        seed=8,
        stage="both",
        extra_args=("train.ppo.max_agent_steps=1024",),
        stage1_extra_args=("train.ppo.max_agent_steps=1500000000",),
        stage2_extra_args=("train.ppo.max_agent_steps=200000000",),
        runtime_profile=modal_train.A100_COMPAT_PROFILE,
        tactile=True,
        pointcloud_points=200,
        rl_variant=modal_train.RL_VARIANT_PPO_CONTACT_AUX,
        auto_eval=True,
        auto_eval_num_seeds=3,
    )

    assert result is fake_call
    assert calls == [
        (
            "demo",
            8,
            "both",
            ("train.ppo.max_agent_steps=1024",),
            ("train.ppo.max_agent_steps=1500000000",),
            ("train.ppo.max_agent_steps=200000000",),
            True,
            200,
            modal_train.RL_VARIANT_PPO_CONTACT_AUX,
            True,
            3,
        )
    ]


def test_run_stage_pipeline_runs_stages_and_eval_sequentially(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "_run_stage",
        lambda stage, run_name, seed=0, extra_args=(), tactile=False: calls.append(
            ("stage", stage, run_name, seed, extra_args)
        ),
    )
    monkeypatch.setattr(
        modal_train,
        "_run_eval_sweep",
        lambda manifest, output_dir="", dry_run=False, wandb_name="", wandb_group="eval", auto_run_name="", tactile_args=(), pointcloud_points=1024, num_seeds=5: calls.append(
            ("eval", manifest, output_dir, dry_run, wandb_name, wandb_group, auto_run_name, tactile_args, pointcloud_points, num_seeds)
        ),
    )

    modal_train._run_stage_pipeline(
        "demo",
        seed=9,
        stage="both",
        extra_args=("train.ppo.max_agent_steps=1024",),
        stage1_extra_args=("train.ppo.max_agent_steps=1500000000",),
        stage2_extra_args=("train.ppo.max_agent_steps=200000000",),
        runtime_profile=modal_train.A100_COMPAT_PROFILE,
        tactile=True,
        pointcloud_points=100,
        rl_variant=modal_train.RL_VARIANT_PPO_RECURRENT,
        auto_eval=True,
        auto_eval_num_seeds=7,
    )

    recurrent_args = (
        "train.ppo.recurrent_obs=True",
        "train.ppo.recurrent_obs_seq_len=3",
        "train.ppo.recurrent_hidden_size=128",
        "task.env.hora.nPointCloudPts=100",
        "train.ppo.n_pointcloud_pts=100",
    )
    assert calls == [
        (
            "stage",
            1,
            "demo",
            9,
            (
                "train.ppo.max_agent_steps=1500000000",
                *recurrent_args,
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=False",
            ),
        ),
        (
            "stage",
            2,
            "demo",
            9,
            (
                "train.ppo.max_agent_steps=200000000",
                *recurrent_args,
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=True",
            ),
        ),
        (
            "eval",
            "",
            modal_train.get_auto_eval_output_dir("demo"),
            False,
            modal_train.get_auto_eval_wandb_name("demo"),
            "eval",
            "demo",
            (
                "train.ppo.max_agent_steps=200000000",
                *recurrent_args,
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=True",
            ),
            100,
            7,
        ),
    ]


def test_run_requested_stages_can_dispatch_stage3(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modal_train,
        "train_stage1_a100_compat_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage1", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage2_a100_compat_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage2", run_name, seed, extra_args))),
    )
    monkeypatch.setattr(
        modal_train,
        "train_stage3_a100_compat_remote",
        SimpleNamespace(remote=lambda run_name, seed, extra_args: calls.append(("stage3", run_name, seed, extra_args))),
    )

    modal_train.run_requested_stages(
        "demo",
        seed=8,
        stage="3",
        extra_args=("task.env.hora.useTactileObs=True", "task.env.hora.useTactileHist=True"),
        runtime_profile=modal_train.A100_COMPAT_PROFILE,
    )

    assert calls == [
        (
            "stage3",
            "demo",
            8,
            (
                "task.env.hora.useTactileObs=True",
                "task.env.hora.useTactileHist=True",
                *DEFAULT_POINTCLOUD_ARGS,
            ),
        ),
    ]


def test_main_parses_overrides_before_dispatch(monkeypatch):
    captured = {}

    def fake_run_requested_stages_durable(
        run_name,
        seed=0,
        stage="both",
        extra_args=(),
        stage1_extra_args=(),
        stage2_extra_args=(),
        runtime_profile=modal_train.DEFAULT_RUNTIME_PROFILE,
        tactile=False,
        pointcloud_points=1024,
        rl_variant=modal_train.RL_VARIANT_PPO,
        auto_eval=False,
        auto_eval_num_seeds=modal_train.DEFAULT_AUTO_EVAL_NUM_SEEDS,
    ):
        captured["run_name"] = run_name
        captured["seed"] = seed
        captured["stage"] = stage
        captured["extra_args"] = extra_args
        captured["stage1_extra_args"] = stage1_extra_args
        captured["stage2_extra_args"] = stage2_extra_args
        captured["runtime_profile"] = runtime_profile
        captured["tactile"] = tactile
        captured["pointcloud_points"] = pointcloud_points
        captured["rl_variant"] = rl_variant
        captured["auto_eval"] = auto_eval
        captured["auto_eval_num_seeds"] = auto_eval_num_seeds

    monkeypatch.setattr(modal_train, "run_requested_stages_durable", fake_run_requested_stages_durable)

    modal_train.main(
        run_name="demo",
        seed=5,
        stage="2",
        overrides='task.env.numEnvs=64 "train.notes=hello world"',
        stage1_overrides="train.ppo.max_agent_steps=1500000000",
        stage2_overrides="train.ppo.max_agent_steps=200000000",
        runtime_profile=modal_train.A100_COMPAT_PROFILE,
        tactile=True,
        pointcloud_points=1024,
        auto_eval=True,
        auto_eval_num_seeds=9,
    )

    assert captured == {
        "run_name": "demo",
        "seed": 5,
        "stage": "2",
        "extra_args": ("task.env.numEnvs=64", "train.notes=hello world"),
        "stage1_extra_args": ("train.ppo.max_agent_steps=1500000000",),
        "stage2_extra_args": ("train.ppo.max_agent_steps=200000000",),
        "runtime_profile": modal_train.A100_COMPAT_PROFILE,
        "tactile": True,
        "pointcloud_points": 1024,
        "rl_variant": modal_train.RL_VARIANT_PPO,
        "auto_eval": True,
        "auto_eval_num_seeds": 9,
    }
