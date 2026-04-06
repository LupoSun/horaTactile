import os

from omegaconf import OmegaConf

from hora.utils.checkpoint_utils import get_algo_best_checkpoint_relpath, get_stage_best_checkpoint_relpath
from hora.utils.misc import git_diff_config, git_hash, write_git_diff_patch, write_run_metadata


def test_git_helpers_fall_back_outside_git_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert git_hash() == "nogit"
    assert git_diff_config("./") == ""


def test_write_git_diff_patch_skips_when_git_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    patch_path = tmp_path / "gitdiff.patch"
    assert write_git_diff_patch(patch_path) is False
    assert not patch_path.exists()


def test_write_run_metadata_does_not_crash_without_git(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    write_run_metadata(str(output_dir), OmegaConf.create({"train": {"ppo": {"output_name": "demo"}}}))

    config_files = list(output_dir.glob("config_*_nogit.yaml"))
    assert len(config_files) == 1
    assert config_files[0].read_text()
    assert not (output_dir / "gitdiff.patch").exists()


def test_checkpoint_relpaths_match_stage_specific_artifacts():
    output_name = "AllegroHandHora/demo"
    assert get_stage_best_checkpoint_relpath(output_name, 1) == os.path.join(
        "outputs", output_name, "stage1_nn", "best.pth"
    )
    assert get_stage_best_checkpoint_relpath(output_name, 2) == os.path.join(
        "outputs", output_name, "stage2_nn", "model_best.ckpt"
    )
    assert get_algo_best_checkpoint_relpath(output_name, "PPO") == os.path.join(
        "outputs", output_name, "stage1_nn", "best.pth"
    )
    assert get_algo_best_checkpoint_relpath(output_name, "ProprioAdapt") == os.path.join(
        "outputs", output_name, "stage2_nn", "model_best.ckpt"
    )
