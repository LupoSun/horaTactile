# 2026-04-15 — Visualization Setup for Policy Playback

## Summary
- Confirmed that policy visualization works locally on WSL, but only when the Isaac Gym WSL environment script is sourced before launching the viewer.
- The failure mode without that setup is misleading at first glance: the script starts, then Isaac Gym fails to create the PhysX CUDA context, falls back to CPU, and finally crashes because this task expects GPU state tensors.

## Working Command Sequence
Use this exact order:

```bash
cd ~/horaTactile
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
bash scripts/vis_s1.sh hora_v0.0.2
```

For Stage 2 visualization:

```bash
cd ~/horaTactile
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
bash scripts/vis_s2.sh hora_v0.0.2
```

## What Went Wrong Before
Running only:

```bash
source .venv/bin/activate
bash scripts/vis_s1.sh hora_v0.0.2
```

was not enough.

The key error chain was:
- `libcuda.so!`
- `Failed to create a PhysX CUDA Context Manager. Falling back to CPU.`
- `Must enable GPU pipeline to use state tensors`

The HORA task uses Isaac Gym GPU state tensors, so once PhysX falls back to CPU, the environment crashes during state refresh.

## Why `isaac_wsl_env.sh` Matters
`scripts/isaac_wsl_env.sh` sets the WSL-specific environment needed for Isaac Gym viewer rendering, especially around Vulkan / GPU viewer setup.

Without it:
- the viewer path is unstable
- CUDA / PhysX initialization can fail even though PyTorch still sees the GPU

## Existing Visualization Entry Points
The repo already has the playback scripts:
- `scripts/vis_s1.sh`
- `scripts/vis_s2.sh`

These launch:
- `headless=False`
- `task.env.numEnvs=1`
- the appropriate checkpoint under `outputs/AllegroHandHora/<run>/`

## Recommended Habit
For any local visualization on WSL, always do:

```bash
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
```

before running viewer-based scripts.
