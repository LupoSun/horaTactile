# Full Tactile Flag Semantics

## Why This Entry Exists

This entry records the updated meaning of Modal `--tactile`. It supersedes the older
notes that described `--tactile` as Stage-2 tactile-history-only behavior, while keeping
those older entries unchanged for historical context.

## New `--tactile` Contract

`--tactile` now enables tactile signals in both places needed for the current tactile
training setup:

- direct tactile actor/PPO observations in Stage 1
- direct tactile control-policy observations in Stage 2
- tactile adaptation history in Stage 2

Concretely, `modal_train.py` now expands `--tactile` to these Stage 1 overrides:

```text
task.env.hora.useTactileObs=True
task.env.hora.useTactileHist=False
```

For Stage 2, Stage 3, and Stage 2 eval, it expands to:

```text
task.env.hora.useTactileObs=True
task.env.hora.useTactileHist=True
```

## Tensor Shapes

Without `--tactile`, the actor observation remains:

```text
obs: 96D
3 x 32D frames
each 32D frame = 16 normalized joint positions + 16 current PD targets
```

With `--tactile`, the actor observation becomes:

```text
obs: 132D
3 x 44D frames
each 44D frame = 16 normalized joint positions + 16 current PD targets + 12 tactile force values
```

The 12D tactile vector is:

```text
4 fingertips x 3 xyz net contact force
```

The tactile force is scaled by `/10` and clipped to `[-1, 1]`.

For shape-aware runs, the privileged/extrinsic vector remains 40D:

```text
z_phys: 8D
z_shape: 32D
z: 40D
```

So the actor input widths are:

```text
without --tactile: 96D obs + 40D z = 136D
with --tactile:    132D obs + 40D z = 172D
```

Stage 2 adaptation history is:

```text
without --tactile: proprio_hist = 30 x 32
with --tactile:    proprio_hist = 30 x 44
```

## Compatibility Note

A tactile Stage 2 run must load a tactile Stage 1 checkpoint. Mixing a Stage 1 checkpoint
trained without direct tactile actor observations with a Stage 2 run that uses direct
tactile actor observations will produce an actor weight shape mismatch.

# Commands Run

```bash
conda activate hora2
export WANDB_API_KEY=your_key_here
```

## Baseline
```bash
modal run --detach modal_train.py::main \
--run-name fullbaseline_04291133 \
--runtime-profile a100_compat \
--stage both \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"

modal run --detach modal_train.py::stage2_eval \
--run-name fullbaseline_04291133 \
--runtime-profile a100_compat \
--pointcloud-points 200 \
--num-seeds 3
```

## Full Tactile
Pointcloud + Tactile for hist (stage2) + Tactile in OBS (stage 1 + 2)
```bash
modal run --detach modal_train.py::main \
--run-name fulltactile_04291133 \
--runtime-profile a100_compat \
--stage both \
--tactile \
--auto-eval \
--auto-eval-num-seeds 1 \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"

modal run --detach modal_train.py::stage2_eval \
--run-name fulltactile_04291133 \
--runtime-profile a100_compat \
--tactile \
--pointcloud-points 200 \
--num-seeds 3
```
