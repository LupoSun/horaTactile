# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HORA (In-Hand Object Rotation via Rapid Motor Adaptation) — a reinforcement learning system for dexterous in-hand object manipulation using AllegroHand robots. Built on NVIDIA IsaacGym for GPU-accelerated parallel physics simulation. Fork of [HaozhiQi/hora](https://github.com/HaozhiQi/hora) with tactile sensing extensions.

## Environment Setup

Requires Python 3.8 (IsaacGym constraint), IsaacGym Preview 4.0 specifically (Preview 3 gives different results).

```bash
conda env create -f environment.yml
conda activate hora
# IsaacGym must be installed separately:
cd isaacgym/python && pip install -e .
```

Download grasp pose cache before training:
```bash
gdown 1xqmCDCiZjl2N7ndGsS_ZvnpViU7PH7a3 -O cache/data.zip
unzip cache/data.zip -d cache/
```

## Key Commands

### Training (two-stage pipeline)
```bash
# Stage 1: PPO with privileged object info
scripts/train_s1.sh ${GPU_ID} ${SEED} ${OUTPUT_NAME}
# Stage 2: Proprioceptive adaptation (requires stage 1 checkpoint)
scripts/train_s2.sh ${GPU_ID} ${SEED} ${OUTPUT_NAME}
# For public AllegroHand, append: task=PublicAllegroHandHora
```

### Evaluation and Visualization
```bash
scripts/eval_s1.sh ${GPU_ID} ${RUN_NAME}
scripts/eval_s2.sh ${GPU_ID} ${RUN_NAME}
scripts/vis_s1.sh ${RUN_NAME}    # requires display
scripts/vis_s2.sh ${RUN_NAME}
```

### Hardware Deployment
```bash
python deploy.py  # ROS-based deployment to real AllegroHand
```

All training/eval invokes `train.py` at root, which uses Hydra for config resolution.

## Architecture

### Two-Stage Training Pipeline
- **Stage 1 (PPO)**: Trains actor-critic with privileged object information (ground-truth physical properties). Algorithm: `hora/algo/ppo/ppo.py`.
- **Stage 2 (ProprioAdapt)**: Freezes the stage 1 policy and trains an adaptation module that infers object properties from proprioceptive sensor history. Algorithm: `hora/algo/padapt/padapt.py`. Loads stage 1 checkpoint from `outputs/.../stage1_nn/best.pth`.

### Code Layout
- `train.py` — Entry point. Hydra-configured, dispatches to PPO or ProprioAdapt via `config.train.algo`.
- `deploy.py` — Entry point for real hardware deployment.
- `hora/tasks/` — IsaacGym environments. `allegro_hand_hora.py` is the main rotation task; `allegro_hand_grasp.py` handles grasp generation. Both inherit from `base/vec_task.py`. Task name maps: `AllegroHandHora`, `PublicAllegroHandHora`, `AllegroHandGrasp`, `PublicAllegroHandGrasp`.
- `hora/algo/models/` — Neural network definitions (`models.py`) and normalization (`running_mean_std.py`).
- `hora/algo/ppo/` — PPO implementation with experience buffer (`experience.py`).
- `hora/algo/padapt/` — Proprioceptive adaptation training loop.
- `hora/algo/deploy/` — Hardware deployment player and robot interfaces (`robots/` subdirectory).
- `hora/utils/` — Config reformatting (`reformat.py`) and seed/git utilities (`misc.py`).
- `configs/` — Hydra configs. Top-level `config.yaml` with `task/` and `train/` config groups. Internal vs public AllegroHand variants exist for both task and training configs.

### Config System
Hydra with OmegaConf. Custom resolvers: `eq`, `contains`, `if`, `resolve_default`. Override any config value via CLI: `python train.py task.env.numEnvs=4096 train.ppo.priv_info=True`. Default: 16384 parallel environments on GPU.

### Output Structure
Training outputs go to `outputs/{task}/{run_name}/` with:
- `stage1_nn/` and `stage2_nn/` — checkpoints (including `best.pth`)
- `gitdiff.patch` — git diff at training time
- `config_{date}_{hash}.yaml` — full resolved config snapshot

### Important Constraint
`import isaacgym` must come before `import torch` — IsaacGym's gymtorch extension requires this load order. This is why `train.py` imports isaacgym first and tasks before algo modules.

## Experiments
```
modal run modal_train.py --run-name baseline --runtime-profile a100_compat --stage 2
modal run modal_train.py --run-name naive_tactile --runtime-profile a100_compat --stage 2 --overrides "task.env.hora.useTactileHist=True"

modal run --detach modal_train.py --run-name baseline_s1 --runtime-profile a100_compat --stage 1 
modal run --detach modal_train.py --run-name double_tactile_s1 --runtime-profile a100_compat --stage 1 --overrides "task.env.hora.useTactileObs=True"
modal run --detach modal_train.py --run-name double_tactile_s1 --runtime-profile a100_compat --stage 2 --overrides "task.env.hora.useTactileHist=True task.env.hora.useTactileObs=True"
```