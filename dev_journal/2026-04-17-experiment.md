# 2026-04-17 — BTG_13 Size Sweep, Eval Tooling, and Isaac Gym Recorder

## Summary
- Expanded the BTG_13 custom object from a 2-size check into a reusable multi-size benchmark ladder.
- Added a manifest-driven evaluation sweep runner for comparing baseline Stage 2 against tactile Stage 2 across object sizes.
- Added a summary-and-plot utility for sweep outputs.
- Added an offscreen Isaac Gym recording script that can export GIF/MP4 plus optional per-frame PNGs.
- Completed a release-backed 150-case BTG_13 sweep comparing vanilla HORA Stage 2, Haojun's released tactile Stage 1 checkpoint, and Haojun's released tactile Stage 2 checkpoint.

## Why This Work
After getting custom mesh playback working, the next need was no longer “can the object load?” but:
- can the same geometry be tested at multiple controlled physical sizes?
- can baseline and tactile policies be compared systematically instead of manually?
- can the resulting behavior be rendered reproducibly as media rather than only through the live viewer?

This work was the tooling layer needed to answer those questions cleanly.

## Part 1 — BTG_13 Size Ladder

### Source geometry
The source mesh remains:

- [`Nodes/BTG_13.stl`](/home/lupo/horaTactile/Nodes/BTG_13.stl)

As noted previously, its raw extents strongly suggest millimeter units, so all exported simulation assets continue to use:

```bash
--export-unit-scale 0.001
```

to convert mm -> m.

### Current benchmark-sized ladder
The intended size ladder is now 5 evenly spaced points from original size to the benchmark mean size, using linear interpolation on the longest bbox edge.

The 5 current BTG_13 variants are:

| Variant | Object type | Longest edge (m) |
|---|---|---|
| Original | `custom_btg13_original` | `0.281975` |
| 25% toward mean | `custom_btg13_lerp_25` | `0.233259` |
| 50% toward mean | `custom_btg13_lerp_50` | `0.184543` |
| 75% toward mean | `custom_btg13_lerp_75` | `0.135827` |
| Mean-sized | `custom_btg13_mean` | `0.087111` |

Generated bundles:
- [`assets/custom/btg13_original/BTG_13`](/home/lupo/horaTactile/assets/custom/btg13_original/BTG_13)
- [`assets/custom/btg13_lerp_25/BTG_13`](/home/lupo/horaTactile/assets/custom/btg13_lerp_25/BTG_13)
- [`assets/custom/btg13_lerp_50/BTG_13`](/home/lupo/horaTactile/assets/custom/btg13_lerp_50/BTG_13)
- [`assets/custom/btg13_lerp_75/BTG_13`](/home/lupo/horaTactile/assets/custom/btg13_lerp_75/BTG_13)
- [`assets/custom/btg13_mean/BTG_13`](/home/lupo/horaTactile/assets/custom/btg13_mean/BTG_13)

Approximate exported bbox extents:

```text
btg13_original: [0.280799, 0.281975, 0.129306]
btg13_lerp_25: [0.232287, 0.233259, 0.106966]
btg13_lerp_50: [0.183774, 0.184543, 0.084626]
btg13_lerp_75: [0.135261, 0.135827, 0.062286]
btg13_mean:    [0.086748, 0.087111, 0.039947]
```

### Notes
- Earlier experimental variants such as `btg13_mid`, `btg13_lerp_1_3`, and `btg13_lerp_2_3` were useful while converging on the design, but the intended current ladder is the 5 evenly spaced sizes above.
- Collision generation still falls back to convex hulls in this environment because `coacd` / `vhacdx` are not installed.

## Part 2 — Evaluation Sweep Runner

### New runner
Added:

- [`scripts/eval_object_sweep.py`](/home/lupo/horaTactile/scripts/eval_object_sweep.py)
- [`hora/utils/eval_sweep.py`](/home/lupo/horaTactile/hora/utils/eval_sweep.py)

This runner is manifest-driven and expands a matrix of:

```text
models x objects x seeds
```

into concrete `train.py ... test=True task.on_evaluation=True` subprocess runs.

### What it standardizes
The sweep runner fixes the evaluation setup so model-to-model comparisons are fair:
- `headless=True`
- `test=True`
- `task.on_evaluation=True`
- mass / COM / friction / PD gain randomization disabled
- object scaling randomization disabled
- external random forces disabled

It also exposes:
- `task.env.numEnvs`
- `task.maxEvaluateEnvs`

through the manifest, so the number of parallel envs and total evaluation sample count can be tuned without editing code.

### Evaluation budget config
To support reusable sweeps, the task config now has:

- [`configs/task/AllegroHandHora.yaml`](/home/lupo/horaTactile/configs/task/AllegroHandHora.yaml)

with:

```yaml
maxEvaluateEnvs: 500000
```

and the task reads that dynamically instead of hardcoding the evaluation budget.

### Current BTG13 comparison template
Added:

- [`configs/eval_sweeps/btg13_stage2_size_comparison.template.json`](/home/lupo/horaTactile/configs/eval_sweeps/btg13_stage2_size_comparison.template.json)

Current template settings:
- models:
  - baseline Stage 2
  - tactile Stage 2
- sizes:
  - the 5-point BTG_13 ladder above
- seeds:
  - `42, 43, 44`
- eval sample count per run:
  - `20000`
- parallel env count:
  - `4096`

That expands to:

```text
2 models x 5 sizes x 3 seeds = 30 eval cases
```

Dry-run verification confirmed the current template prepares 30 cases successfully.

### Release-backed comparison manifest
After locating Haojun's GitHub release, the reported comparison was run with:

- [`configs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds.json`](/home/lupo/horaTactile/configs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds.json)

That manifest compares:
- `vanilla_stage2`
- `tactile_stage1_release`
- `tactile_stage2_release`

across the same 5 BTG_13 sizes and 10 seeds:

```text
3 models x 5 sizes x 10 seeds = 150 eval cases
```

The release checkpoints were extracted locally under:

- [`outputs/AllegroHandHora/double_tactile_s1_2`](/home/lupo/horaTactile/outputs/AllegroHandHora/double_tactile_s1_2)

## Part 3 — Sweep Result Visualization

### New plot/summarize layer
Added:

- [`scripts/plot_eval_sweep.py`](/home/lupo/horaTactile/scripts/plot_eval_sweep.py)
- [`hora/utils/eval_plots.py`](/home/lupo/horaTactile/hora/utils/eval_plots.py)

Before this, the sweep runner only produced raw:
- `results.json`
- `results.csv`

Now sweep outputs can also be summarized into:
- `summary.csv`
- `summary.md`
- PNG plots per metric

Supported default metrics:
- `rotate_reward`
- `reward`
- `eps_length`
- `lin_vel_x100`
- `command_torque`

### Output behavior
Given a completed sweep directory:

```bash
python scripts/plot_eval_sweep.py outputs/eval_sweeps/<run_dir>
```

the script writes:

```text
plots/
  summary.csv
  summary.md
  rotate_reward.png
  reward.png
  eps_length.png
  lin_vel_x100.png
  command_torque.png
```

The summarizer groups rows by:

```text
model_name x object_name
```

and aggregates over seeds with mean/std so the size-performance curve is visible directly.

### Dependency work
To support this plotting path, `matplotlib` was added to:

- [`requirements.txt`](/home/lupo/horaTactile/requirements.txt)

and installed into the project `.venv`.

## Part 4 — Isaac Gym Recorder

### New recorder
Added:

- [`scripts/record_policy.py`](/home/lupo/horaTactile/scripts/record_policy.py)
- [`hora/utils/recording.py`](/home/lupo/horaTactile/hora/utils/recording.py)

The recorder is built around Isaac Gym camera sensors rather than screen-capturing the viewer.

That means it can:
- run headless
- render from a fixed offscreen camera
- save deterministic media from chosen checkpoints
- work on arbitrary object variants by passing `task.env.object.type`

### What it records
The script can write:
- animated GIF
- MP4
- optional per-frame PNGs

It supports:
- `--run-name` or explicit `--checkpoint`
- `--stage 1|2`
- `--object-type`
- `--tactile`
- `--steps`
- `--frame-every`
- `--fps`
- `--width` / `--height`
- `--camera-position`
- `--camera-target`
- `--frames-dir`

### Example
Example Stage 2 BTG recording:

```bash
python scripts/record_policy.py \
  --run-name hora_v0.0.2 \
  --stage 2 \
  --object-type custom_btg13_lerp_50 \
  --steps 400 \
  --output outputs/recordings/btg13_lerp_50.mp4
```

### Implementation notes
The recorder:
- composes the standard Hydra config instead of inventing a separate env path
- creates an offscreen camera sensor with Isaac Gym
- advances the policy in inference mode for a finite number of steps
- captures RGB frames through `get_camera_image(..., IMAGE_COLOR)`
- converts Isaac Gym RGBA image layout into regular RGB arrays

It also sets:

```text
WANDB_MODE=disabled
TORCH_EXTENSIONS_DIR=<repo>/.torch_extensions
```

to avoid unnecessary logging side effects and to keep Isaac Gym / Torch extension compilation in a writable project-local cache.

### Dependencies
To support media export, these were added to:

- [`requirements.txt`](/home/lupo/horaTactile/requirements.txt)

and installed into `.venv`:
- `imageio`
- `imageio-ffmpeg`

## Validation

### Automated tests
Added and passed:
- [`tests/test_eval_sweep.py`](/home/lupo/horaTactile/tests/test_eval_sweep.py)
- [`tests/test_eval_plots.py`](/home/lupo/horaTactile/tests/test_eval_plots.py)
- [`tests/test_recording_utils.py`](/home/lupo/horaTactile/tests/test_recording_utils.py)

Relevant local checks that passed:
- `pytest -q tests/test_eval_sweep.py`
- `pytest -q tests/test_eval_plots.py tests/test_eval_sweep.py`
- `pytest -q tests/test_recording_utils.py`
- dry-run sweep expansion for the BTG_13 template
- plot smoke test on a synthetic sweep result table

### Practical note
The recorder utilities and dependencies are in place, but a full live Isaac Gym recording run was not completed inside this sandboxed environment. The remaining real-world validation step is to run `scripts/record_policy.py` on the actual machine where viewer playback already works.

## Part 5 — Release Checkpoint Compatibility Work

### Why this needed a bit more implementation work
Running the finished BTG13 comparison against Haojun's released tactile checkpoints exposed two compatibility mismatches that were not obvious from the earlier tooling work alone.

First, the release checkpoints were not using the exact same tactile convention as the simplified current branch:
- released tactile Stage 1 expected a `132`-dim observation stream
- released tactile Stage 2 expected that same `132`-dim observation stream plus a `44`-dim tactile history

To support that cleanly, the task config/path was extended with:
- `task.env.hora.useTactileObs`
- `task.env.hora.useTactileHist`

and the environment now switches observation dimensionality accordingly instead of assuming a single tactile mode.

Second, the custom BTG assets exposed a reset-path assumption in [`allegro_hand_hora.py`](/home/lupo/horaTactile/hora/tasks/allegro_hand_hora.py): even with `randomizeScale=False`, reset still expected cached grasp initializations keyed by scale. To keep the BTG size ladder fixed while still using the existing grasp cache mechanism, a fixed-scale grasp-init override was added through:

- `task.env.randomization.graspInitScale`

This lets evaluation use:
- fixed custom object geometry size for the BTG ladder
- fixed grasp-cache initialization scale (`0.8`)

without reintroducing random object scale variation during evaluation.

### Why this matters for the final sweep
This compatibility work was what made the final reported comparison possible:
- Haojun's released Stage 1 and Stage 2 checkpoints now load on the current branch
- the BTG13 size ladder can be held fixed during evaluation
- tactile observation vs tactile history can be toggled independently to match the release expectations

With those fixes in place, the final reported experiment could focus on the full 10-seed comparison instead of per-checkpoint compatibility debugging.

## Part 6 — Final 10-Seed BTG13 Sweep

### Run configuration
The reported comparison used:
- 3 models
- 5 BTG_13 sizes
- 10 seeds
- `5000` evaluation episodes per case
- `4096` parallel envs per case
- mass / COM / friction / PD / external force randomization disabled
- `jointNoiseScale=0.0`

The controlling manifest is:

- [`configs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds.json`](/home/lupo/horaTactile/configs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds.json)

using seeds:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

That expands to:

```text
3 models x 5 sizes x 10 seeds = 150 eval cases
```

### 10-seed outputs
The completed run is:

- [`outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353)

Generated summary artifacts:
- [`summary.csv`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/summary.csv)
- [`summary.md`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/summary.md)
- [`rotate_reward.png`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/rotate_reward.png)
- [`reward.png`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/reward.png)
- [`eps_length.png`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/eps_length.png)
- [`lin_vel_x100.png`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/lin_vel_x100.png)
- [`command_torque.png`](/home/lupo/horaTactile/outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/command_torque.png)

All 150 cases completed successfully with `0` errors.

### Rotate-reward table — 10 seeds

| Size variant | Longest edge (m) | Vanilla Stage 2 | Tactile Stage 1 release | Tactile Stage 2 release |
| --- | --- | --- | --- | --- |
| `btg13_original` | `0.281975` | `-1.4120` | `-1.4360` | `-1.4480` |
| `btg13_lerp_25` | `0.233259` | `-0.3360` | `-0.4650` | `-0.4530` |
| `btg13_lerp_50` | `0.184543` | `3.0510` | `-0.4260` | `-0.4300` |
| `btg13_lerp_75` | `0.135827` | `7.1570` | `0.3630` | `0.3320` |
| `btg13_mean` | `0.087111` | `94.4280` | `1.2150` | `1.1550` |

### Plot gallery — 10-seed sweep

#### Rotate reward
![10-seed rotate reward](../outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/rotate_reward.png)

#### Reward
![10-seed reward](../outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/reward.png)

#### Episode length
![10-seed episode length](../outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/eps_length.png)

#### Linear velocity
![10-seed linear velocity](../outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/lin_vel_x100.png)

#### Command torque
![10-seed command torque](../outputs/eval_sweeps/btg13_vanilla_stage1_stage2_release_10seeds_20260417_230353/plots/command_torque.png)

### Stronger final interpretation
After the 10-seed run, the conservative conclusion is:

```text
On the BTG13 size ladder, the released tactile checkpoints consistently underperform vanilla HORA Stage 2, and that conclusion is stable across 10 seeds.
```

## Takeaway
By the end of this work, the repo has moved from:

```text
single custom mesh playback experiment
```

to a more complete evaluation stack:

- a controlled 5-size BTG_13 geometry ladder
- a reusable model x size x seed evaluation sweep runner
- a summary/plot layer for sweep outputs
- a headless Isaac Gym recorder for image/video export
- a release-compatible tactile evaluation path for Haojun's published checkpoints
- one completed 150-case benchmark run with plots, tables, and aggregated results

This is no longer just tooling for a comparison. It is now a completed BTG13 benchmark pass with reusable artifacts, embedded plots, and a stable result: the released tactile checkpoints did not outperform vanilla HORA on this custom-object size ladder.
