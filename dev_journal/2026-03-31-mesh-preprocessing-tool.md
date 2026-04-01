# 2026-03-31 — Mesh Preprocessing Tool

## Summary

This entry now combines:
- the **implemented** mesh preprocessing work
- the **paper-grounded integration plan** for how those assets plug into HORA-style and RotateIt-style training

Only the preprocessing tool itself is implemented today.

The downstream integration work is **not** done yet, and this note now reflects that explicitly instead of mixing finished and planned work together.

## What Was Built

`tools/mesh/preprocess.py` — a new CLI pipeline added in this repo that converts Rhino-exported meshes into IsaacGym-ready assets plus point clouds for future geometry-aware training.

This is not something HORA shipped with. It was added here specifically for the Rhino-object benchmark and fills a real gap in the original HORA codebase:
- the existing repo only uses simple simulation assets such as cylinders and cuboids
- there was no mesh ingestion pipeline for Rhino-modeled objects
- there was no automatic way to generate both physics assets and PointNet-ready point clouds from the same source mesh

## Why This Matters

This project needs one shared geometry pipeline for two benchmark paths:

- **Path A**: HORA-style benchmark on custom objects where the policy gets no explicit geometry input
- **Path B**: RotateIt-style oracle benchmark where object shape is included in privileged information through a point-cloud encoder

Both paths start from the same step:

`Rhino mesh -> cleaned simulation asset + sampled point cloud`

The preprocessing tool is that shared step.

## Paper Grounding

### Path A vs Path B

| | Path A | Path B |
|---|---|---|
| **Paper** | HORA — Qi et al., CoRL 2022 | RotateIt — Qi et al., CoRL 2023 |
| **Training objects** | Cylinders only | Many real meshes |
| **Geometry in privileged info** | None | Explicit point-cloud embedding |
| **Stage 1 privileged signal** | Physics-focused low-dimensional vector | Physics embedding + shape embedding |
| **Stage 2 input** | Proprioception history | Vision + touch + proprioception in the paper |
| **What this repo would test** | Can touch/proprio alone generalize to new shapes? | How much does explicit shape help the oracle? |

### HORA grounding

HORA's Stage 1 oracle receives a privileged vector built from object state and physics, then compresses it into an 8D latent `z_t`. Stage 2 learns to infer that latent from a history of proprioceptive interaction, without any explicit geometry input.

That is the key scientific baseline for this repo:
- if Path A works well on novel Rhino meshes, then touch/proprio dynamics alone may be enough to adapt across shapes

### RotateIt grounding

RotateIt extends the oracle by adding **explicit shape** through a PointNet-style encoder over sampled object surface points.

Conceptually:
- `z_phys`: learned embedding of object physics and pose
- `z_shape`: learned embedding of point-cloud geometry
- `z_t = concat(z_phys, z_shape)`

That is the geometry-aware comparison point:
- if Path B clearly outperforms Path A on the same custom objects, then explicit shape carries useful information that touch/proprio alone does not recover

## Implemented: Mesh Preprocessing Pipeline

### Inputs

The tool takes Rhino-exported meshes such as:
- `.obj`
- `.stl`
- `.ply`
- scene/multi-body exports that `trimesh` can merge into a single mesh

Typical source location:

```text
assets/custom/raw/
```

### Outputs

For each input mesh, the tool writes:

```text
assets/custom/{object_name}/
  visual.obj
  collision.obj
  {object_name}.urdf
  pointcloud_100.npy
  pointcloud_1024.npy
  metadata.json
```

### What the tool does

Per mesh:

1. Load and clean with `trimesh`
2. Merge scene geometry if needed
3. Remove duplicate / degenerate faces and repair normals / winding
4. Center the mesh at the centroid
5. Normalize to a unit bounding sphere
6. Export `visual.obj`
7. Build collision geometry with:
   - `coacd` if available
   - otherwise trimesh convex decomposition
   - otherwise convex hull fallback
8. Compute inertia:
   - exact from volume if watertight
   - bounding-box approximation otherwise
9. Generate a URDF with repo-root-relative mesh paths
10. Sample surface point clouds and save them as `.npy`
11. Save metadata for downstream simulation/debugging

### Output conventions

The point clouds are stored in **unit-sphere normalized coordinates**.

That means:
- the mesh has been centered at the origin
- the mesh has been scaled so its bounding sphere radius is `1`
- the original scale factor is preserved in `metadata.json`

This is appropriate for future PointNet-style shape encoding, because the point clouds represent shape independent of world pose.

### Metadata currently saved

The tool currently saves:
- `canonical_scale_factor`
- `centroid_offset`
- original and normalized bounding boxes
- volume when available
- surface area
- face / vertex counts
- watertightness flag
- mass
- inertia
- point-cloud resolutions written
- URDF filename

## CLI Usage

```bash
# Single mesh
python tools/mesh/preprocess.py path/to/shape.obj

# Batch
python tools/mesh/preprocess.py assets/custom/raw/*.obj

# Custom output directory
python tools/mesh/preprocess.py shape.obj --output-dir assets/my_objects

# Skip convex decomposition
python tools/mesh/preprocess.py shape.obj --skip-decomposition

# Custom point-cloud resolutions
python tools/mesh/preprocess.py shape.obj --n-points 100 512 2048

# Custom object mass
python tools/mesh/preprocess.py shape.obj --mass 0.05

# Verbose logs
python tools/mesh/preprocess.py shape.obj --verbose
```

## Rhino Export Notes

Before running the tool:
- export each object from Rhino as `.obj` or `.stl`
- keep track of export units
- prefer manifold / watertight meshes when possible
- export normals for `.obj`

The tool can handle non-watertight geometry, but inertia becomes approximate in that case.

## Current Repo Status

### Implemented

- mesh preprocessing CLI exists at `tools/mesh/preprocess.py`
- asset folders are generated under `assets/custom/{name}/`
- URDF generation is automated
- collision meshes are generated
- point clouds are generated
- metadata is generated

### Not implemented yet

- no `custom` object branch is wired into the HORA task config / asset loading path
- no eval script yet loads `assets/custom/*/` automatically
- no PointNet encoder exists in the actor-critic model
- no point cloud is currently fed into privileged info
- `configs/train/AllegroHandHora.yaml` still uses `priv_info_dim: 9`
- the current training code is still HORA-style, not RotateIt-style

So the preprocessing step is done, but the benchmark paths that consume those assets are still future work.

## Next Integration Work

## Path A — Custom objects at eval time

This is the smaller next step and remains fully compatible with the HORA paper framing.

Goal:
- keep training exactly as HORA does now
- evaluate on custom Rhino meshes without giving the policy explicit geometry

What still needs to be added:
- a new custom object type in task config, e.g. `task.env.object.type=custom`
- task-side asset discovery for `assets/custom/*/*.urdf`
- an eval script that points the environment at those generated URDFs

What should stay unchanged for Path A:
- Stage 1 algorithm
- privileged info semantics
- Stage 2 architecture
- no point-cloud input to policy

Why this benchmark is important:
- it directly tests whether tactile/proprio adaptation learned from cylinder interactions transfers to novel shapes

## Path B — Geometry-aware oracle policy

This remains a research extension, not implemented code.

Goal:
- extend the oracle with explicit shape information following RotateIt's basic idea

Minimum required changes:
- load `pointcloud_100.npy` per object
- add a PointNet-style encoder in the policy model
- append or concatenate the learned shape embedding into privileged information
- train Stage 1 on custom meshes rather than cylinder-only assets

Current code gap relative to that goal:
- no point cloud buffer in the environment
- no point-cloud field in `obs_dict`
- no PointNet module in `hora/algo/models/models.py`
- no expanded privileged-info config path yet

### Practical note on paper fidelity

The original RotateIt paper goes beyond "just add PointNet":
- it uses a richer privileged decomposition than HORA
- it includes harder multi-axis rotation settings
- its Stage 2 model is a visuotactile transformer, not the current HORA adaptation model

For this repo, the likely first useful comparison is not a full RotateIt reproduction. It is:
- keep the existing HORA-style training stack as much as possible
- add only the geometry-aware oracle signal needed to measure whether explicit shape helps

That keeps the benchmark focused and makes the ablation interpretable.

## Suggested Benchmark Framing

Once Path A and Path B exist, the clean comparison is:

- **Path A**: same custom objects, no explicit geometry in the policy
- **Path B**: same custom objects, same simulator/randomization family, but with explicit geometry in the oracle

Interpretation:
- if **Path A ~= Path B**, touch/proprio may already recover enough shape information
- if **Path B >> Path A**, explicit geometry is filling a real information gap

That is the main scientific reason the mesh preprocessing tool was worth building first.

## Dependencies

Runtime dependencies for the tool:
- `trimesh`
- `numpy`

Recommended optional dependency:
- `coacd`

`scipy` is useful in the broader geometry stack, but the preprocessing tool itself is fundamentally built around `trimesh` plus optional `coacd`.

## Notes

- `coacd` is preferred for complex concave shapes and thin features
- point clouds are sampled from the normalized mesh and are intended for shape encoding, not pose encoding
- the tool writes repo-root-relative mesh paths into the URDF so they align with Isaac Gym asset loading conventions
- the generated folder layout is already suitable for a future `assets/custom/*/` glob-based object loader
