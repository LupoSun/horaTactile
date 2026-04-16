# 2026-04-15 — BTG_13 Custom Object Visualization Experiment

## Summary
- Turned `Nodes/BTG_13.stl` into two Isaac Gym custom-object asset bundles:
  - one at source size with a millimeter-to-meter conversion
  - one uniformly scaled to match the mean size of the default HORA demonstration objects
- Verified that the repo can now discover and load custom mesh assets through `task.env.object.type=custom_<subset>`.
- Added a dedicated Stage 2 visualization script so custom objects can be played back without editing the stock viewer scripts.

## Why This Experiment
The mesh preprocessing pipeline was already generating simulation-ready bundles, but the repo still needed a concrete end-to-end check on a real STL from `Nodes/`.

`BTG_13.stl` was used as the first custom-object probe for two questions:
- can we load a custom Rhino-style mesh into the current HORA environment and visualize policy playback on it?
- what is a sensible physical size for that object relative to the benchmark objects used in the paper setting?

## What Was Implemented

### Custom object asset loading
Added a shared object-catalog helper:

- [`hora/utils/object_assets.py`](/home/lupo/horaTactile/hora/utils/object_assets.py)

and wired it into:

- [`hora/tasks/allegro_hand_hora.py`](/home/lupo/horaTactile/hora/tasks/allegro_hand_hora.py)

This extends object discovery beyond the original hardcoded primitives and now supports:
- `cylinder_*`
- `cuboid_*`
- `custom_*`

Custom assets are discovered under:

```text
assets/custom/<subset>/*.urdf
assets/custom/<subset>/*/*.urdf
```

So if a bundle is placed under `assets/custom/btg13_mean/BTG_13/BTG_13.urdf`, it can be loaded with:

```text
task.env.object.type=custom_btg13_mean
```

### Mesh preprocessing update
Extended:

- [`tools/mesh/preprocess.py`](/home/lupo/horaTactile/tools/mesh/preprocess.py)

with:
- `--export-unit-scale`

This was important because the STL appears to be authored in millimeters, while Isaac/HORA assets should be in meters.

For this experiment:
- `--export-unit-scale 0.001` was used to convert mm -> m

### Custom-object visualization entry point
Added:

- [`scripts/vis_object_s2.sh`](/home/lupo/horaTactile/scripts/vis_object_s2.sh)

Usage:

```bash
bash scripts/vis_object_s2.sh <run_name> <object_type>
```

This avoids hand-editing `vis_s2.sh` whenever a custom object needs to be tested.

## BTG_13 Asset Generation

### Source mesh inspection
`BTG_13.stl` has the following raw axis-aligned bounding-box extents:

```text
[280.799438, 281.975403, 129.305725]
```

This strongly suggests the source units are millimeters.

### Variant 1 — Original physical size
Generated with:

```bash
.venv/bin/python tools/mesh/preprocess.py \
  Nodes/BTG_13.stl \
  --output-dir assets/custom/btg13_original \
  --export-unit-scale 0.001
```

Output bundle:

- [`assets/custom/btg13_original/BTG_13`](/home/lupo/horaTactile/assets/custom/btg13_original/BTG_13)

Exported mesh bbox:

```text
[0.280799, 0.281975, 0.129306] m
```

This preserves the source physical size after mm -> m conversion.

### Variant 2 — Mean benchmark size
The default HORA benchmark objects in:
- `assets/cylinder/default/`
- `assets/cuboid/default/`

contain 90 demonstration objects in total.

Their mean longest bbox edge is:

```text
0.0871111111111111 m
```

That value was used as the target size for a paper-like custom-object comparison.

Generated with:

```bash
.venv/bin/python tools/mesh/preprocess.py \
  Nodes/BTG_13.stl \
  --output-dir assets/custom/btg13_mean \
  --export-unit-scale 0.001 \
  --target-bbox-size 0.0871111111111111
```

Output bundle:

- [`assets/custom/btg13_mean/BTG_13`](/home/lupo/horaTactile/assets/custom/btg13_mean/BTG_13)

Exported mesh bbox:

```text
[0.086748, 0.087111, 0.039947] m
```

This is the better “benchmark-sized” version for actual evaluation.

## Size Comparison

### Original-size BTG_13
- longest edge: `0.281975 m`

### Mean-sized BTG_13
- longest edge: `0.087111 m`

### Ratio
- original / mean-sized: `3.23696x`

So the original-size object is much larger than the default benchmark family and should be treated mainly as a stress test or visualization sanity check, not the first serious manipulation target.

## Visualization Commands
This assumes the WSL Isaac Gym setup documented in:

- [`2026-04-15-visualization-setup.md`](/home/lupo/horaTactile/dev_journal/2026-04-15-visualization-setup.md)

Use:

```bash
cd ~/horaTactile
source scripts/isaac_wsl_env.sh
source .venv/bin/activate

bash scripts/vis_object_s2.sh hora_v0.0.2 custom_btg13_original
bash scripts/vis_object_s2.sh hora_v0.0.2 custom_btg13_mean
```

These use the existing Stage 2 checkpoint under:

- [`outputs/AllegroHandHora/hora_v0.0.2/stage2_nn/model_last.ckpt`](/home/lupo/horaTactile/outputs/AllegroHandHora/hora_v0.0.2/stage2_nn/model_last.ckpt)

## Issues Encountered

### Collision decomposition fallback
The preprocessing run completed, but true convex decomposition was not available in this environment because:
- `coacd` was not installed
- `vhacdx` was not installed

So collision generation fell back to:
- convex hull

That is good enough for loading and viewer playback, but not ideal for contact fidelity.

### Custom assets were not previously wired into HORA
Before this experiment, the repo’s environment logic only handled the built-in primitive families plus `simple_tennis_ball`.

The custom mesh pipeline existed, but there was no automatic path from:

```text
assets/custom/... -> HORA object loader
```

This experiment closed that loop.

## Validation
Local checks completed:

- generated both BTG_13 asset bundles successfully
- verified `metadata.json` sizes for both variants
- verified the expected files exist:
  - `visual.obj`
  - `collision.obj`
  - `BTG_13.urdf`
  - `pointcloud_100.npy`
  - `pointcloud_1024.npy`
  - `metadata.json`

Test coverage added/updated around this path:
- [`tests/test_object_assets.py`](/home/lupo/horaTactile/tests/test_object_assets.py)
- [`tests/test_mesh_preprocess.py`](/home/lupo/horaTactile/tests/test_mesh_preprocess.py)

## Takeaway
The repo can now:
- preprocess arbitrary STL meshes into HORA-compatible bundles
- preserve original physical size when source units are known
- rescale custom objects to match the benchmark object family
- load those custom objects directly inside the HORA task
- visualize Stage 2 policy playback on them through a dedicated script

For BTG_13 specifically:
- `custom_btg13_mean` is the appropriate first evaluation target
- `custom_btg13_original` is useful as a large-object stress test
