# Geometry Integration Plan — Custom Rhino Meshes

## Status Note

Parts of this original plan are now implemented.

Implemented:
- the mesh preprocessing pipeline in [`tools/mesh/preprocess.py`](/home/lupo/horaTactile/tools/mesh/preprocess.py)
- bbox-based mesh scaling in [`tools/mesh/scale_by_bbox.py`](/home/lupo/horaTactile/tools/mesh/scale_by_bbox.py)
- source-unit conversion for exported sim meshes through `--export-unit-scale`
- custom-object asset discovery in [`hora/utils/object_assets.py`](/home/lupo/horaTactile/hora/utils/object_assets.py)
- HORA-side custom object loading in [`hora/tasks/allegro_hand_hora.py`](/home/lupo/horaTactile/hora/tasks/allegro_hand_hora.py)
- Stage 2 custom-object playback in [`scripts/vis_object_s2.sh`](/home/lupo/horaTactile/scripts/vis_object_s2.sh)

Still planned / not implemented:
- geometry-aware privileged information with a PointNet-style encoder
- true RotateIt-style oracle training
- visuotactile Stage 2 architecture

This document remains the research/planning note. For the implemented pipeline and the first BTG_13 asset/playback experiment, see:
- [`2026-03-31-mesh-preprocessing-tool.md`](/home/lupo/horaTactile/dev_journal/2026-03-31-mesh-preprocessing-tool.md)
- [`2026-04-15-btg13-custom-object-visualization.md`](/home/lupo/horaTactile/dev_journal/2026-04-15-btg13-custom-object-visualization.md)

## Context

This project benchmarks in-hand object rotation on custom-shaped objects modeled in Rhino.
The pipeline needs to support two paths simultaneously:

- **Path A** — Objects in simulation only (IsaacGym physics), policy trained on cylinders, tests generalization of tactile adaptation to novel shapes. No geometry fed to the policy.
- **Path B** — Geometry-aware oracle policy (à la RotateIt), where a PointNet embedding of the object's point cloud is included in the privileged information vector and used to train a shape-aware Stage 1 policy.

Both paths share the same **mesh preprocessing pipeline** (Rhino → simulation-ready assets). They diverge only in how (or whether) geometry enters the policy's privileged info.

---

## Path A vs Path B — Paper Grounding

### Side-by-side

| | **Path A** | **Path B** |
|---|---|---|
| **Paper** | HORA — Qi et al., CoRL 2022 | RotateIt — Qi et al., CoRL 2023 |
| **Training objects** | Cylinders only (9 URDFs, scale ∈ [0.70, 0.86]) | Hundreds of meshes (EGAD, YCB, Google Scanned, ContactDB) |
| **Rotation axes** | z-axis only | x, y, z (all three principal axes) |
| **Privileged info `e_t`** | 9D: position (3) + scale (1) + mass (1) + friction (1) + COM (3) | 17D raw: physics 7D + pose 10D → encoded to `z_phys` (8D); shape: PointNet(100 pts) → `z_shape` (32D) |
| **Extrinsics `z_t`** | 8D (MLP encoder `[256, 128, 8]`) | 40D = `z_phys` (8D) + `z_shape` (32D) |
| **Geometry in priv info** | None — scale scalar is a cylinder-specific proxy | Explicit: PointNet on `N_p = 100` surface points → `c_p = 32D` |
| **Stage 2 architecture** | MLP on 30-step proprioception history | Visuotactile **Transformer** on depth + tactile + proprioception history |
| **Stage 2 sensor inputs** | Joint positions + actions only | Object depth image (60×60) + fingertip contact locations + joint positions + actions |
| **Sim environments** | 16,384 (1 GPU) | 32,768 (4 GPUs) |
| **Randomisation: scale** | [0.70, 0.86] | [0.46, 0.68] |
| **Randomisation: mass** | [0.01, 0.25] kg | [0.01, 0.25] kg |
| **Randomisation: friction** | [0.3, 3.0] | [0.3, 3.0] |
| **Randomisation: COM** | ±1 cm per axis | ±1 cm per axis |
| **Restitution** | Not randomised | Randomised (range not published) |

### Path A — HORA (2022) in depth

The oracle (Stage 1) receives `e_t ∈ R^9` which is projected through a 3-layer MLP
`µ: R^9 → R^8` to produce `z_t`. The policy input is `o_t = (q_{t-2:t}, a_{t-3:t-1}) ∈ R^96`
concatenated with `z_t`.

Stage 2 learns `φ: (q_{t-30:t}, a_{t-31:t-1}) → ẑ_t` by supervised regression against `z_t`
produced by `µ` — no simulator access needed during Stage 2 training. The paper shows
two dimensions of the learned `z_t` spontaneously correlate with object **diameter**
(`z_{t,0}`) and **mass** (`z_{t,2}`), demonstrating that shape information is implicitly
encoded through contact dynamics without any explicit geometry input.

Real-world results: successfully rotates 30+ objects (4.5–7.5 cm diameter, 5–200 g)
over the z-axis with zero real-world fine-tuning.

### Path B — RotateIt (2023) in depth

The key advance is splitting the privileged vector into physics+pose and shape:

```
z_phys (8D)  ←  MLP( mass(1) + COM(3) + friction(1) + scale(1) + restitution(1)
                     + position(3) + orientation_quat(4) + angular_velocity(3) )
                                         = MLP( 17D → 8D )

z_shape (32D) ←  PointNet( sample 100 pts from mesh → shared per-point MLP + max-pool → 32D )

z_t (40D)    =  concat( z_phys, z_shape )
```

The PointNet is a standard per-point MLP with global max-pooling, trained
**end-to-end** with the PPO policy — not pretrained separately.

**Performance impact (Table 1, z-axis, same training conditions):**

| Method | RotR ↑ | TTF ↑ | RotP ↓ |
|---|---|---|---|
| HORA baseline | 99.83 | 0.60 | 0.39 |
| + pose, no shape | 129.38 | 0.75 | 0.29 |
| Full oracle (+ shape) | 140.90 | 0.82 | 0.27 |

On x/y-axis rotation, HORA scores 79–82 RotR while the full oracle reaches 118–125 —
a gap of roughly **+50%** attributable primarily to the shape embedding on harder axes.

Stage 2 uses a **Transformer** that ingests a history of:
- Object depth images (60×60) encoded by a 4-layer ConvNet → 32D `f^depth`
- Fingertip contact locations from tactile sensors → 32D `f^touch`
- Proprioception + actions (same as HORA Stage 2)

The transformer is trained to minimise `‖z_t − ẑ_t‖²` across the full 40D vector,
meaning it must learn to estimate shape from vision+touch simultaneously.

### What the benchmark measures

Running both paths on the **same custom Rhino objects** isolates exactly one variable:
whether having explicit shape in the oracle policy (and thus in Stage 2's regression
target) improves performance when Stage 2 still has only tactile + proprioceptive inputs.

- If **Path A ≈ Path B**: tactile sensing alone is sufficient to adapt across geometries —
  the core claim of the horaTactile project.
- If **Path B >> Path A**: explicit geometry provides an information gap that touch alone
  cannot close, motivating adding a depth camera to the adaptation module.

---

## Part 1 — Mesh Preprocessing Pipeline (shared by both paths)

### Input
- Rhino exports: `.obj` or `.stl` files, one per object, placed in `assets/custom/raw/`

### Output (per object)
```
assets/custom/
  {object_name}/
    visual.obj           ← cleaned mesh for rendering
    collision.obj        ← convex decomposition for physics (V-HACD)
    {object_name}.urdf   ← IsaacGym-loadable wrapper
    pointcloud_100.npy   ← 100 surface-sampled points, shape (100, 3), normalised to unit sphere
    pointcloud_1024.npy  ← full-res version for offline analysis
    metadata.json        ← bounding box, volume, surface area, canonical scale
```

### Script: `tools/mesh/preprocess.py`

**Steps per mesh:**

1. **Load & clean** (trimesh)
   - Remove duplicate vertices, fix winding, fill small holes
   - Centre at origin (subtract centroid)
   - Normalise to unit bounding sphere (divide by max radius) — store scale factor in `metadata.json`

   The current implementation also supports:
   - source-unit conversion for exported sim meshes, e.g. `--export-unit-scale 0.001` for mm -> m
   - optional bbox-target scaling for exported sim meshes while keeping point clouds unit-normalised

2. **Convex decomposition** for collision
   - Run **V-HACD** (`pybullet.vhacd` or `coacd`) to generate `collision.obj`
   - IsaacGym needs convex collision geometry for stable simulation

3. **URDF generation**
   ```xml
   <robot name="{object_name}">
     <link name="object">
       <visual>
         <geometry><mesh filename="visual.obj" scale="1 1 1"/></geometry>
       </visual>
       <collision>
         <geometry><mesh filename="collision.obj" scale="1 1 1"/></geometry>
       </collision>
       <inertial>
         <mass value="0.1"/>
         <inertia ixx="..." .../>   <!-- computed from mesh volume -->
       </inertial>
     </link>
   </robot>
   ```

4. **Point cloud sampling** (trimesh `sample_surface`)
   - Sample 1024 points uniformly from surface → save as `pointcloud_1024.npy`
   - Subsample to 100 points → save as `pointcloud_100.npy`
   - Points already in normalised (unit sphere) coordinates from step 1

5. **Metadata export** (`metadata.json`)
   ```json
   {
     "name": "object_name",
     "canonical_scale": 0.073,
     "bbox": [0.12, 0.08, 0.15],
     "volume_m3": 0.00042,
     "surface_area_m2": 0.021
   }
   ```

### CLI usage
```bash
python tools/mesh/preprocess.py assets/custom/raw/my_shape.obj
# or batch:
python tools/mesh/preprocess.py assets/custom/raw/*.obj
```

### New dependencies to add to `requirements.txt`
```
trimesh
scipy          # already likely present
coacd          # convex decomposition (pip install coacd)
```

---

## Part 2 — Path A: Cylinder-trained policy, Rhino objects at eval

### What changes
Only evaluation / asset loading — **no model changes**.

#### What is implemented now
Custom assets are now discovered through:

- [`hora/utils/object_assets.py`](/home/lupo/horaTactile/hora/utils/object_assets.py)

and consumed in:

- [`hora/tasks/allegro_hand_hora.py`](/home/lupo/horaTactile/hora/tasks/allegro_hand_hora.py)

The current selector format is:

```text
task.env.object.type=custom_<subset>
```

where assets are loaded from:

```text
assets/custom/<subset>/*.urdf
assets/custom/<subset>/*/*.urdf
```

#### Current playback script

- [`scripts/vis_object_s2.sh`](/home/lupo/horaTactile/scripts/vis_object_s2.sh)

Example:

```bash
bash scripts/vis_object_s2.sh hora_v0.0.2 custom_btg13_mean
```

#### What still needs to be added

- a matching Stage 1 custom-object playback helper
- a dedicated eval script that batches over custom subsets
- benchmark bookkeeping around which subset and size convention is being used

### What you learn
Whether tactile + proprioceptive adaptation (Stage 2) generalises from cylinders to arbitrary geometries **without** ever seeing those geometries during training. This is the core scientific question of `horaTactile`.

---

## Part 3 — Path B: Geometry-aware privileged info (RotateIt approach)

### Overview
Extend the privileged info vector from **9D → 40D** by adding a PointNet-encoded shape embedding (32D). Requires training on the custom meshes, not just cylinders.

### 3a — New config fields (`configs/train/AllegroHandHora.yaml`)
```yaml
ppo:
  priv_info_dim: 40          # was 9
  shape_embed_dim: 32        # PointNet output dimension
  n_pointcloud_pts: 100      # points per object

network:
  pointnet:
    units: [64, 128, 32]    # per-point MLP layers (last = shape_embed_dim)
```

### 3b — Environment changes (`hora/tasks/allegro_hand_hora.py`)

**Init:** Load `pointcloud_100.npy` for each object asset at env creation. Store as a dict `{asset_id: tensor(100, 3)}` on GPU.

**`compute_observations()`:** Append the per-environment point cloud tensor to `obs_dict`:
```python
obs_dict['point_cloud'] = self.point_cloud_buf  # (num_envs, 100, 3)
```

**`_update_priv_buf()`:** Extend priv_info buffer to 40D:
- dims 0–7: existing physics + pose (keep as-is, now also include orientation + angular velocity — see below)
- dims 8–39: PointNet embedding (computed in model, not here)

**Extended physics dims** (upgrade from 9D baseline while we're here):
| Dim | Property | Notes |
|---|---|---|
| 0–2 | Object position | existing |
| 3 | Scale | existing |
| 4 | Mass | existing |
| 5 | Friction | existing |
| 6–8 | COM offset | existing |
| 9–12 | Object orientation (quat) | **new** |
| 13–15 | Angular velocity | **new** |

So physics block = 16D, shape block = 32D → total = **48D** (or keep 9D physics + 32D shape = 41D; decide based on ablation goals).

### 3c — Model changes (`hora/algo/models/models.py`)

Add a `PointNetEncoder` module:
```python
class PointNetEncoder(nn.Module):
    """Shared MLP + max-pool PointNet. Input: (B, N, 3). Output: (B, embed_dim)."""
    def __init__(self, units=[64, 128, 32]):
        ...  # per-point Linear → BN → ELU layers, then max-pool
```

Integrate into `ActorCritic.__init__()`:
```python
self.pointnet = PointNetEncoder(units=cfg['pointnet_units'])
```

In `ActorCritic.forward()` / `_actor_critic()`:
```python
z_shape = self.pointnet(input_dict['point_cloud'])       # (B, 32)
priv_embedding = self.env_mlp(input_dict['priv_info'])   # (B, 8) — existing physics
z = torch.cat([priv_embedding, z_shape], dim=-1)         # (B, 40)
```
Then concatenate `z` into the main actor MLP as before.

**ProprioAdapt (Stage 2):** The adaptation target becomes the full 40D `z_t`. The adaptation module architecture is otherwise unchanged (it learns to predict whatever `z_t` the Stage 1 oracle used).

### 3d — `priv_info_dim` propagation
The config field `ppo.priv_info_dim` is threaded through `net_config` into the model. Change it to 16 (physics only); the 32D shape embedding is added by PointNet separately, not counted in `priv_info_dim`. Keeps the interface clean.

---

## Part 4 — New files summary

| File | Purpose |
|---|---|
| `tools/mesh/preprocess.py` | Rhino mesh → URDF + point clouds + metadata |
| `assets/custom/` | Output directory for processed custom objects |
| `scripts/vis_object_s2.sh` | Visualize Stage 2 policy on custom objects (Path A) |

## Modified files

| File | Change |
|---|---|
| `hora/tasks/allegro_hand_hora.py` | Custom object type loading |
| `hora/utils/object_assets.py` | Primitive/custom asset discovery |
| `tools/mesh/preprocess.py` | Preprocessing, source-unit conversion, bbox-target scaling |
| `scripts/vis_object_s2.sh` | Custom-object Stage 2 playback |
| `requirements.txt` | Add `trimesh`, `coacd` |

---

## Suggested implementation order

1. Completed: `tools/mesh/preprocess.py`
2. Completed: Path A asset discovery + custom object loading
3. Completed: local Stage 2 playback on custom objects
4. Next: Path B model changes (`PointNetEncoder`)
5. Next: Path B env changes (point cloud buffer in `obs_dict`)
6. Next: retrain Stage 1 with geometry-aware priv info on mixed cylinder + custom object set

---

## Open questions

- **Collision fidelity**: V-HACD is fast but approximate. For objects with thin features (handles, fins), consider `coacd` which gives better decompositions.
- **Point cloud normalisation**: Should points be in object-local frame (always unit sphere) or world frame? Object-local is standard for PointNet and avoids leaking pose information into shape.
- **Mixed training (Path B)**: Train on cylinders + custom objects together, or custom objects only? Mixed is safer for the adaptation module's generalisation.
- **Ablation design**: Path A vs Path B results on the same custom object set is the core benchmark comparison this project is set up to run.
