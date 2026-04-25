# Shape-Aware Tactile Cylinder Training

## Goal

Use the new `assets/custom/cylinder_2dcross/` and `assets/custom/cylinder_3dcross/`
objects during Stage 1 and Stage 2 training, mixed with the usual cylinder set, to make
the tactile/proprioceptive adaptation problem see richer contact geometry.

The implementation follows the Qi 2023 oracle-to-sensor-policy methodology, using the
smaller PointNet setup described in the paper, but without the depth/vision branch. In
this repo, the oracle receives privileged object physics, pose, and shape. The Stage 2
policy learns to infer that oracle extrinsic representation from proprioceptive and
tactile history only.

## Training Object Mix

Stage 1 and Stage 2 helpers now default to:

```text
task.env.object.type=cylinder_default+custom_cylinder_2dcross+custom_cylinder_3dcross
task.env.object.sampleProb=[0.34,0.33,0.33]
```

This applies to:

- `scripts/train_s1.sh`
- `scripts/train_s2.sh`
- `modal_train.py` Stage 1 and Stage 2 command builders

The eval and visualization shell helpers were also updated to reconstruct shape-aware
checkpoints with the same object mix and model shape.

The base YAML keeps the shape path disabled for backwards compatibility. The Modal and
shell training helpers opt into this setup by passing the overrides below.

## Privileged Oracle Signal

The new shape-aware path is opt-in through config flags:

```text
task.env.hora.useShapePrivInfo=True
task.env.hora.useExtendedPrivInfo=True
task.env.hora.privInfoDim=17
task.env.hora.nPointCloudPts=100
train.ppo.use_shape_priv_info=True
train.ppo.priv_info_dim=17
train.ppo.n_pointcloud_pts=100
```

The extended raw privileged vector contains:

- existing 9D HORA fields: object position, scale, mass, friction, and COM
- object orientation quaternion
- object angular velocity
- a reserved restitution slot, currently zero because restitution is not randomized here

The model then projects this 17D physics/pose vector to an 8D `z_phys` embedding with a
three-layer privileged MLP `[256, 128, 8]` and ReLU activations.

## Shape Encoding

Custom cylinder-cross assets already include `pointcloud_100.npy` and `pointcloud_1024.npy`
sidecars. The task now
loads the point cloud for each object asset and stores the per-environment tensor in
`obs_dict["point_cloud"]`.

During training/evaluation, it rotates each environment's point cloud by the current object
quaternion before placing it in `obs_dict["point_cloud"]` (see Qi 2023 fig. 2).
Translation is intentionally not applied; object
position remains in the privileged vector.

The usual primitive cylinder URDFs do not have point cloud sidecars, so
`hora.utils.object_assets.load_object_point_cloud()` supports flat, stem-specific
sidecars such as `assets/cylinder/default/0000_pointcloud_100.npy`. Both 100-point and
1024-point sidecars exist for the default cylinder set. They can be regenerated with:

```bash
PYTHONPATH=. python scripts/generate_cylinder_pointclouds.py
PYTHONPATH=. python scripts/generate_cylinder_pointclouds.py --n-points 1024
```

The loader intentionally does not generate missing point clouds during training. If a
sidecar is missing, it raises `FileNotFoundError` and reports the expected filenames.

`hora.algo.models.models.PointNetEncoder` is the smaller Qi 2023-style encoder:

- three per-point MLP layers with hidden units `[32, 32, 32]`
- ReLU activations
- max pooling across points

It outputs a 32D `z_shape` vector.

The default point count is 100 to match the smaller paper setup and avoid the memory
pressure of dense point clouds. Modal exposes `--pointcloud-points`, which can be set to
`100` or `1024`. The equivalent Hydra overrides are:

```text
task.env.hora.nPointCloudPts=1024
train.ppo.n_pointcloud_pts=1024
```

The oracle extrinsic vector is:

```text
z = concat(z_phys, z_shape) = 8D + 32D = 40D
```

Note: an earlier heavier PointNet implementation with transform nets and 1024-channel
layers caused A100 40GB OOM at default HORA environment counts. The current implementation
uses the smaller Qi 2023 per-point MLP and does not need PointNet microbatching.

## Stage 2 Adaptation

`ProprioAdaptTConv` now accepts a configurable output dimension. For shape-aware runs it
predicts the same 40D extrinsic vector used by the oracle policy. The Stage 2 loss remains
the L2 loss between predicted extrinsics and oracle extrinsics, but the target now includes
shape.

No depth, RGB, camera, or visual transformer input was added. The Stage 2 input remains
proprioceptive history plus optional tactile history, matching the current tactile learning
setup.

## Files Changed

- `hora/tasks/allegro_hand_hora.py`
- `hora/algo/models/models.py`
- `hora/algo/ppo/ppo.py`
- `hora/algo/ppo/experience.py`
- `hora/algo/padapt/padapt.py`
- `hora/utils/object_assets.py`
- `hora/utils/eval_sweep.py`
- `hora/utils/recording.py`
- `configs/task/AllegroHandHora.yaml`
- `configs/train/AllegroHandHora.yaml`
- `.gitignore`
- `modal_train.py`
- `scripts/generate_cylinder_pointclouds.py`
- `scripts/train_s1.sh`
- `scripts/train_s2.sh`
- `scripts/eval_s1.sh`
- `scripts/eval_s2.sh`
- `scripts/vis_s1.sh`
- `scripts/vis_s2.sh`
- `scripts/viz_pointcloud.py`
- `tests/test_modal_train.py`
- `tests/test_object_assets.py`
- `tests/test_wandb_utils.py`

## Verification

Completed:

```text
python -m py_compile ...modified Python files...
PYTHONPATH=. pytest -q tests/test_object_assets.py tests/test_eval_sweep.py tests/test_recording_utils.py
```

The dependency-light tests passed: `17 passed`.

A direct PyTorch smoke test also verified that:

- shape-aware Stage 1 returns 40D oracle extrinsics
- shape-aware Stage 1 backprop reaches the lightweight PointNet with both 100 and 1024 points
- shape-aware Stage 2 returns 40D predicted extrinsics and 40D oracle targets

## Point Cloud Visualization

Use `scripts/viz_pointcloud.py` for quick visual checks of the canonical point cloud
sidecars. It defaults to 100 points and can also display 1024-point sidecars:

```bash
python scripts/viz_pointcloud.py assets/cylinder/default
python scripts/viz_pointcloud.py assets/custom/cylinder_2dcross
python scripts/viz_pointcloud.py assets/custom/cylinder_3dcross
python scripts/viz_pointcloud.py assets/cylinder/default assets/custom/cylinder_2dcross assets/custom/cylinder_3dcross
python scripts/viz_pointcloud.py --n-points 1024 assets/cylinder/default
```

The script discovers both custom object sidecars named like `pointcloud_100.npy` and
primitive-cylinder sidecars named like `0000_pointcloud_100.npy`. It uses a shared
x/y/z range and equal box aspect across all displayed subplots so object size and shape
are visually comparable.

## Experiments Ran

```bash
conda activate hora2
export WANDB_API_KEY=your_key_here

modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_04251206 \
--runtime-profile a100_compat \
--stage 1 \
--pointcloud-points 100
```
