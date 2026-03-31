# 2026-03-31 — Modal Cloud Training Integration

## What was done

### Research
- Investigated options for running the two-stage HORA training pipeline on [Modal](https://modal.com)
- Key finding: IsaacGym's `vec_task.py` automatically sets `graphics_device_id=-1` when `headless=True` and no camera sensors are used, which disables Vulkan entirely — meaning Modal's CUDA-only containers work without any graphics driver workarounds
- Confirmed via Modal docs that their GPU containers (A10, A100, H100, etc.) expose full CUDA but have no documented Vulkan/EGL support — a non-issue given the above

### Logging — TensorBoard → wandb
Replaced `tensorboardX` with `wandb` in both training algorithms:
- `hora/algo/ppo/ppo.py` — `write_stats()` and episode reward logging in `train()`
- `hora/algo/padapt/padapt.py` — `log_tensorboard()` renamed to `log_wandb()`

Both algorithms now call `wandb.init()` with the full Hydra config serialized via `OmegaConf.to_container()`, giving automatic hyperparameter tracking in the wandb dashboard.

### Modal harness — `modal_train.py`
New file at repo root (branch: `modal_integration`, commit: `61f1bfc`).

Key design decisions:
- **Image**: `nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04` + miniconda + Python 3.8 + IsaacGym Preview 4.0 downloaded via the existing NVIDIA gdown link (`1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9`)
- **Volume**: persistent `modal.Volume` at `/vol` for `outputs/`, `cache/`, and `wandb/` data
- **Symlink trick**: before each run, `outputs/` and `cache/` inside the project dir are symlinked to `/vol`, so `train.py`'s relative output paths land on the volume without any changes to training code
- **Overwrite guard**: pre-checks for existing checkpoints and raises early rather than hitting `train.py`'s interactive `input()` prompt
- **Stage chaining**: `main()` entrypoint runs stage 1 then stage 2 by default; `--stage 1` or `--stage 2` to run individually

### Usage
```bash
# One-time: populate grasp cache on volume
modal run modal_train.py::setup_cache_remote

# Train both stages
modal run modal_train.py --run-name my_exp

# Train a single stage
modal run modal_train.py --run-name my_exp --stage 1
modal run modal_train.py --run-name my_exp --stage 2
```

## Files changed
| File | Change |
|---|---|
| `modal_train.py` | Created |
| `hora/algo/ppo/ppo.py` | TensorBoard → wandb |
| `hora/algo/padapt/padapt.py` | TensorBoard → wandb |
| `requirements.txt` | `tensorboard`/`tensorboardx` → `wandb` |
