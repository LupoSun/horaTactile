import numpy as np

from hora.utils.recording import (
    build_recording_overrides,
    default_recording_path,
    resolve_checkpoint_and_output_name,
    rgba_image_to_rgb_array,
)


def test_resolve_checkpoint_and_output_name_prefers_explicit_checkpoint():
    checkpoint, output_name = resolve_checkpoint_and_output_name(
        run_name="ignored",
        stage=2,
        checkpoint="outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt",
    )
    assert checkpoint == "outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt"
    assert output_name == "AllegroHandHora/demo"


def test_resolve_checkpoint_and_output_name_builds_stage_best_path():
    checkpoint, output_name = resolve_checkpoint_and_output_name(run_name="hora_v0.0.2", stage=1)
    assert checkpoint == "outputs/AllegroHandHora/hora_v0.0.2/stage1_nn/best.pth"
    assert output_name == "AllegroHandHora/hora_v0.0.2"


def test_default_recording_path_uses_stage_and_object_type():
    path = default_recording_path("AllegroHandHora/hora_v0.0.2", 2, "custom_btg13_mean")
    assert path.as_posix() == "outputs/recordings/hora_v0.0.2__stage2__custom_btg13_mean.gif"


def test_build_recording_overrides_enables_camera_and_disables_eval_randomization():
    overrides = build_recording_overrides(
        output_name="AllegroHandHora/demo",
        checkpoint="outputs/AllegroHandHora/demo/stage2_nn/model_best.ckpt",
        stage=2,
        object_type="custom_btg13_mean",
        use_tactile=True,
        num_envs=1,
        extra_overrides=("task.env.reset_height_threshold=0.6",),
    )
    assert "task.enableCameraSensors=True" in overrides
    assert "task.env.randomization.randomizeScale=False" in overrides
    assert "task.env.randomization.jointNoiseScale=0.0" in overrides
    assert "train.algo=ProprioAdapt" in overrides
    assert "task.env.hora.useTactile=True" in overrides
    assert overrides[-1] == "task.env.reset_height_threshold=0.6"


def test_rgba_image_to_rgb_array_accepts_flat_and_packed_images():
    flat = np.arange(2 * 3 * 4, dtype=np.uint8)
    rgb_from_flat = rgba_image_to_rgb_array(flat, width=3, height=2)
    assert rgb_from_flat.shape == (2, 3, 3)
    assert np.array_equal(rgb_from_flat[0, 0], np.array([0, 1, 2], dtype=np.uint8))

    packed = flat.reshape(2, 12)
    rgb_from_packed = rgba_image_to_rgb_array(packed, width=3, height=2)
    assert np.array_equal(rgb_from_packed, rgb_from_flat)
