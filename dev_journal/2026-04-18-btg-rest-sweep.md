# 2026-04-18 — Remaining BTG Nodes Sweep

## Goal
- Apply the same 5-size ladder evaluation used for `BTG_13` to the remaining `BTG_*.stl` meshes in [`Nodes/`](/home/lupo/horaTactile/Nodes).
- Compare the same 3 checkpoints:
  - vanilla HORA Stage 2
  - Haojun tactile Stage 1 release
  - Haojun tactile Stage 2 release
- Run the long sweep in detached `tmux` so the job survives terminal disconnects.

## Planned Setup
- Mesh source set: remaining `BTG_*.stl` files in [`Nodes/`](/home/lupo/horaTactile/Nodes), excluding `BTG_13` which already has its own completed sweep.
- Size ladder per mesh:
  - `original`
  - `lerp_25`
  - `lerp_50`
  - `lerp_75`
  - `mean`
- Evaluation protocol:
  - 10 seeds
  - 3 model variants
  - fixed-size evaluation with scale randomization disabled
  - detached `tmux` launch

## Files
- Prep script: [`prepare_btg_nodes_sweep.py`](/home/lupo/horaTactile/scripts/prepare_btg_nodes_sweep.py)
- tmux launcher: [`launch_eval_sweep_tmux.sh`](/home/lupo/horaTactile/scripts/launch_eval_sweep_tmux.sh)

## Launch Record
- Manifest:
  - [`btg_nodes_rest_vanilla_stage1_stage2_release_10seeds.json`](/home/lupo/horaTactile/configs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds.json)
- tmux session:
  - `btg_nodes_rest_sweep`
- Output directory:
  - [`btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807`](/home/lupo/horaTactile/outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807)
- Launcher log:
  - [`tmux_launcher.log`](/home/lupo/horaTactile/outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/tmux_launcher.log)

## Current Status
- Asset prep completed for the remaining 12 BTG meshes:
  - `12 meshes x 5 sizes = 60 object variants`
- Sweep expansion:
  - `3 models x 60 objects x 10 seeds = 1800 cases`
- A smoke eval on `custom_btg1_mean` completed successfully before launch, confirming that the new non-BTG13 assets load through the real evaluation path.
- The detached `tmux` run completed successfully:
  - `1800 / 1800` cases completed
  - `0` errors
  - final case: `tactile_stage2_release__btg9_mean__seed51`

## Detached Monitoring
- Attach:
  - `tmux attach -t btg_nodes_rest_sweep`
- Follow the outer launcher log:
  - `tail -f outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/tmux_launcher.log`

The `tmux` session exited normally after the run completed.

## Summary Artifacts
- [`results.csv`](/home/lupo/horaTactile/outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/results.csv)
- [`results.json`](/home/lupo/horaTactile/outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/results.json)
- [`summary.csv`](/home/lupo/horaTactile/outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/plots/summary.csv)
- [`summary.md`](/home/lupo/horaTactile/outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/plots/summary.md)

## Plot Gallery

### Rotate Reward
![rotate reward](../outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/plots/rotate_reward.png)

### Total Reward
![reward](../outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/plots/reward.png)

### Episode Length
![episode length](../outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/plots/eps_length.png)

### Linear Velocity
![linear velocity](../outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/plots/lin_vel_x100.png)

### Command Torque
![command torque](../outputs/eval_sweeps/btg_nodes_rest_vanilla_stage1_stage2_release_10seeds_20260418_125807/plots/command_torque.png)

## Aggregate Findings

### Rotate reward averaged by size variant

| Size variant | Vanilla Stage 2 | Tactile Stage 1 release | Tactile Stage 2 release |
| --- | ---: | ---: | ---: |
| `original` | `3.882` | `-0.374` | `-0.368` |
| `lerp_25` | `4.117` | `-0.096` | `-0.087` |
| `lerp_50` | `8.020` | `0.084` | `0.075` |
| `lerp_75` | `21.666` | `0.504` | `0.477` |
| `mean` | `49.787` | `1.086` | `1.052` |

Across the 60 object-size conditions, vanilla Stage 2 had the highest rotate reward in `56 / 60` conditions.

The 4 non-vanilla wins were all low-performance edge cases:
- `btg10_lerp_25`: tactile Stage 1 was `-0.148`, vanilla was `-0.149`
- `btg2_mean`: tactile Stage 1 was `0.090`, vanilla was `-0.057`
- `btg6_original`: tactile Stage 1 was `-0.242`, vanilla was `-0.404`
- `btg7_original`: tactile Stage 2 was `0.106`, vanilla was `0.008`

Those exceptions do not indicate a strong tactile advantage; they are near-zero or failing regimes.

### Strongest vanilla transfer cases at mean size

| Object | Vanilla rotate reward | Vanilla reward | Episode length |
| --- | ---: | ---: | ---: |
| `btg9_mean` | `176.125` | `123.618` | `416.454` |
| `btg8_mean` | `159.586` | `112.243` | `374.573` |
| `btg10_mean` | `137.923` | `98.539` | `325.747` |
| `btg12_mean` | `36.969` | `23.335` | `107.835` |
| `btg5_mean` | `24.636` | `13.909` | `84.595` |

The result reinforces the same size trend from the `BTG_13` sweep:
- smaller, benchmark-mean-sized objects are much more likely to rotate successfully
- larger original/intermediate sizes are often dominated by early drops, instability, or short episodes

### Tactile Stage 2 vs tactile Stage 1
Tactile Stage 2 did not show a meaningful aggregate improvement over tactile Stage 1.

Average `tactile_stage2 - tactile_stage1` rotate-reward deltas by size:
- `original`: `+0.006`
- `lerp_25`: `+0.009`
- `lerp_50`: `-0.009`
- `lerp_75`: `-0.027`
- `mean`: `-0.034`

The deltas are tiny relative to the vanilla-vs-tactile gap, and tactile Stage 2 was never better than tactile Stage 1 on the `lerp_75` or `mean` groups.

## Interpretation
The broader remaining-BTG sweep supports the same conclusion as the earlier `BTG_13` experiment:

```text
On this custom BTG geometry family, vanilla HORA Stage 2 transfers substantially better than the released tactile checkpoints.
```

The tactile checkpoints load and run correctly, including the release-specific tactile observation/history dimensions, but they do not improve object-rotation transfer on this benchmark. The evidence is now broader than one mesh: it covers 12 additional BTG geometries, 5 sizes per geometry, and 10 seeds per condition.

## Issues Encountered Along The Way
- The first detached launcher attempt failed because `scripts/isaac_wsl_env.sh` was sourced inside a subshell in the tmux launcher, so the exported `LD_LIBRARY_PATH` did not persist into `python`.
- That caused Isaac Gym to fall back to CPU with:
  - `Failed to create a PhysX CUDA Context Manager. Falling back to CPU.`
- The launcher was patched to source the WSL env script in the current shell context, and the sweep was relaunched successfully.

## Notes
- This journal is intended to hold the broader post-BTG13 benchmark record once the remaining-node sweep is launched and completed.
