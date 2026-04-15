# 2026-03-31 — Modal Cloud Training Integration

## Summary

This entry replaces the original "first pass" notes for the Modal integration with the final implementation state as of March 31, 2026 (America/Los_Angeles), including the follow-up fixes and the live smoke-test results.

The integration now works end to end on Modal, including:
- volume-backed cache setup
- stage 1 PPO training
- stage 2 proprio adaptation training
- offline W&B logging by default in unauthenticated remote runs
- Git-safe startup when the container does not have `.git/`

The earlier notes in this file were outdated in a few important ways:
- the final image is **not** the original CUDA 10.2 + Miniconda design
- the final harness does **not** copy `.netrc` into the image
- the final interface supports explicit `--overrides`
- the final implementation includes additional fixes outside `modal_train.py`
- the integration has now been smoke-tested live on Modal with a cheap GPU (`T4`)

## Key Research Result

The core research conclusion held up and is now captured in code and tests:
- Isaac Gym's base task logic disables graphics by setting `graphics_device_id=-1` when `headless=True` and camera sensors are disabled.
- That means Modal's CUDA GPU containers can run this workload without Vulkan / EGL setup.

This logic is now factored into `hora/utils/graphics.py` and covered by tests, instead of existing only as an implementation detail inside `vec_task.py`.

## Final Implementation

### 1. Modal harness

`modal_train.py` is now a real production harness rather than a first draft.

Current design:
- **App**: `modal.App("hora-train")`
- **Volume**: `modal.Volume.from_name("hora-volume", create_if_missing=True)`
- **Project mount/copy**: repo copied into `/root/project`, excluding generated and unnecessary paths like `outputs/`, `cache/`, `.git/`, `.venv/`, `.pytest_cache/`, `.codex/`, and local `isaacgym/`
- **Environment**:
  - `PYTHONPATH=/root/project`
  - `PYTHONUNBUFFERED=1`
  - `WANDB_DIR=/vol/wandb`
- **Default GPU**: `A100`, overridable with `MODAL_GPU`
- **Default base image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`, overridable with `MODAL_BASE_IMAGE`

Current image build strategy:
- Modal runtime Python: **3.11** via `modal.Image.from_registry(..., add_python="3.11")`
- Training/runtime Python: **Ubuntu system Python 3.8** at `/usr/bin/python3`
- PyTorch stack:
  - `torch==2.1.2+cu118`
  - `torchvision==0.16.2+cu118`
  - `torchaudio==2.1.2+cu118`
- Isaac Gym Preview 4 pulled from the existing Google Drive file ID
- Isaac Gym patched during image build with:
  - `sed -i 's/dtype=np.float/dtype=float/' /opt/isaacgym/python/isaacgym/torch_utils.py`

The Python split is intentional:
- Modal's current runner requires Python 3.10+
- Isaac Gym Preview 4 still wants Python 3.8
- So the container runs Modal itself on 3.11, while `train.py` and cache setup run on `/usr/bin/python3`

### 2. Stage orchestration and CLI

The harness now supports:

```bash
# One-time: populate grasp cache on the volume
modal run modal_train.py::setup_cache_remote

# Train both stages
modal run modal_train.py --run-name my_exp

# Train a single stage
modal run modal_train.py --run-name my_exp --stage 1
modal run modal_train.py --run-name my_exp --stage 2

# Pass Hydra overrides explicitly
modal run modal_train.py --run-name my_exp --overrides "task.env.numEnvs=4096 train.ppo.max_agent_steps=1024"
```

Important final behavior:
- `main()` runs stage 1 then stage 2 by default
- `--stage 1` and `--stage 2` are supported
- `--overrides` is parsed with `shlex.split()` and forwarded into `train.py`
- stage 1 and stage 2 command construction is now test-covered

### 3. Volume-backed outputs and cache

Before each training run:
- `/root/project/outputs -> /vol/outputs`
- `/root/project/cache -> /vol/cache`

This keeps the original relative paths used by the training code while persisting artifacts on the Modal volume.

The cache setup logic is now stricter than the original draft:
- it derives the expected cache filenames from `configs/task/AllegroHandHora.yaml`
- it only treats cache as "ready" if the full expected internal grasp-cache set exists
- it no longer treats "any `.npy` file exists" as success

### 4. Stage-specific checkpoint handling

Checkpoint names are now explicit and centralized in `hora/utils/checkpoint_utils.py`:
- stage 1 / `PPO`: `outputs/<run>/stage1_nn/best.pth`
- stage 2 / `ProprioAdapt`: `outputs/<run>/stage2_nn/model_best.ckpt`

This is used in both the training entrypoint and the Modal harness so the behavior is consistent.

The harness now:
- blocks overwriting an existing best checkpoint before launching training
- verifies stage 1 output exists before stage 2 starts
- avoids falling into `train.py`'s interactive overwrite prompt

## Logging Changes

TensorBoard logging was replaced with W&B logging in both trainers:
- `hora/algo/ppo/ppo.py`
- `hora/algo/padapt/padapt.py`

The final implementation uses a shared helper in `hora/utils/wandb_utils.py`:
- serializes the full Hydra config with `OmegaConf.to_container(..., resolve=True)`
- uses `group="stage1"` for PPO
- uses `group="stage2"` for ProprioAdapt
- resolves mode with:
  - `WANDB_MODE` if explicitly set
  - otherwise `online` if `WANDB_API_KEY` exists
  - otherwise `offline`

This fixed an earlier regression where W&B initialization was unconditional and could fail or block in non-interactive environments.

Important operational note:
- local `wandb login` is not enough to make remote Modal runs sync online
- the remote function only gets W&B credentials if `WANDB_API_KEY` is present locally and forwarded as a Modal secret
- otherwise Modal runs log offline to `/vol/wandb`

## Training Startup / Reproducibility Fixes

The first Modal implementation exposed a hidden assumption in the original training code: startup expected a Git checkout to be available.

That has now been fixed.

`hora/utils/misc.py` now behaves safely when `.git/` is missing:
- `git_hash()` returns `"nogit"` if Git metadata is unavailable
- `git_diff_config()` returns `""` on failure
- `write_git_diff_patch(...)` only writes `gitdiff.patch` if `git diff` succeeds

`train.py` now:
- writes run metadata through `write_run_metadata(...)`
- uses stage-specific checkpoint resolution via `get_algo_best_checkpoint_relpath(...)`

This makes the training entrypoint safe to run from a repo copy inside Modal even when `.git/` is excluded from the image.

## Stage 2 Training Fix

`ProprioAdapt.train()` originally used a hardcoded `1e9` loop bound.

That is now fixed:
- stage 2 respects `train.ppo.max_agent_steps`

This matters for both testing and operations:
- smoke tests can terminate quickly
- stage 2 behavior is now consistent with stage 1

## Test Coverage Added

Local regression coverage was added around the integration work.

New/updated test infrastructure:
- `requirements-dev.txt` with `pytest`
- `pytest.ini` with:
  - `testpaths = tests`
  - `norecursedirs = .venv isaacgym outputs cache`

New tests:
- `tests/test_modal_train.py`
  - override parsing
  - expected cache filenames
  - cache completeness behavior
  - symlink setup
  - stage-specific checkpoint behavior
  - stage command generation
  - stage chaining
- `tests/test_wandb_utils.py`
  - W&B mode resolution
  - config serialization
- `tests/test_training_metadata.py`
  - Git-less metadata behavior
- `tests/test_vec_task.py`
  - graphics-device resolution under headless/no-camera conditions

## Live Modal Smoke Tests

Live cloud smoke tests were run on **March 31, 2026** using a **T4** (`MODAL_GPU=T4`).

### Cache setup

Command:

```bash
MODAL_GPU=T4 uvx --with omegaconf modal run modal_train.py::setup_cache_remote
```

Result:
- succeeded
- populated `/vol/cache`
- extracted both `internal_*` and `public_*` grasp-cache files from the downloaded archive

Second run:
- succeeded
- printed `Cache already populated at /vol/cache, skipping download.`

### Stage 1 smoke run

Command:

```bash
MODAL_GPU=T4 uvx --with omegaconf modal run modal_train.py \
  --run-name smoke_20260331_t4_stage1b \
  --stage 1 \
  --overrides "task.env.numEnvs=64 train.ppo.horizon_length=8 train.ppo.minibatch_size=512 train.ppo.mini_epochs=1 train.ppo.max_agent_steps=1024 train.ppo.save_frequency=1"
```

Result:
- succeeded
- Isaac Gym imported and built `gymtorch`
- environment created on `cuda:0`
- W&B initialized in offline mode
- training exited cleanly on the short `max_agent_steps=1024` cap
- outputs written under:
  - `/vol/outputs/AllegroHandHora/smoke_20260331_t4_stage1b/`

### Stage 2 smoke run

Command:

```bash
MODAL_GPU=T4 uvx --with omegaconf modal run modal_train.py \
  --run-name smoke_20260331_t4_stage1b \
  --stage 2 \
  --overrides "task.env.numEnvs=64 train.ppo.max_agent_steps=1024"
```

Result:
- succeeded
- found the stage 1 checkpoint on the volume
- launched `ProprioAdapt`
- W&B initialized in offline mode
- exited cleanly on the short `max_agent_steps=1024` cap

## Follow-up GPU Compatibility Matrix

Additional Modal compatibility testing was run on **April 15, 2026** to understand why the original default `A100` path was unstable while `T4` worked.

### Final observed behavior

| GPU / Profile | Stack | Result |
|---|---|---|
| `T4` / `t4_stable` | `torch 2.1.2+cu118`, CUDA 11.8 | Works |
| `A100-40GB` / `a100_probe` | `torch 2.1.2+cu118`, CUDA 11.8 | Fails |
| `A100-40GB` / `a100_compat` | `torch 1.13.1+cu117`, CUDA 11.7 | Works |
| `H100 80GB` / `h100_probe` | `torch 2.1.2+cu118`, CUDA 11.8 | Works |
| `H100 80GB` / `h100_stable` | `torch 2.1.2+cu118`, CUDA 11.8 | Works |

### What this means

- The failure is **not** caused by the custom mesh pipeline; the compatibility tests used the stock `cylinder_default` object from `modal_train.py`.
- The failure is **not** a generic “powerful GPU” issue; H100 works on the current stack.
- The evidence points to an **A100-specific compatibility problem** in the combination of:
  - Isaac Gym Preview 4
  - GPU PhysX
  - the newer `torch 2.1.2+cu118` Modal image

In the failing `a100_probe` run, the A100 reproduced the crash with:
- `CUDA_LAUNCH_BLOCKING=1`
- explicit `A100-40GB`
- low-level NVIDIA `Xid 31` MMU fault logging before the Python-side illegal memory access surfaced

That made the problem much more likely to be a runtime / kernel-path issue than a bug in the HORA reward logic.

### Runtime profiles added

`modal_train.py` now carries multiple explicit runtime profiles instead of relying on a single generic GPU default:

- `t4_stable`
  - validated baseline on T4
- `a100_probe`
  - current image on explicit `A100-40GB`
  - includes `CUDA_LAUNCH_BLOCKING=1` for debugging
- `a100_compat`
  - A100 fallback image using `torch 1.13.1+cu117`
- `h100_probe`
  - current image on explicit `H100!`
  - includes `CUDA_LAUNCH_BLOCKING=1` for debugging
- `h100_stable`
  - recommended Hopper production path
  - same current working stack as `h100_probe`, but without the debug slowdown
- `h100_compat`
  - conservative Hopper fallback image using the older stack

### Current recommendation

For real training:
- use `h100_stable` on H100
- use `a100_compat` on A100
- keep `a100_probe` and `h100_probe` for diagnostics only

## Issues Encountered And Resolved

This integration only stabilized after a number of real issues showed up during review and smoke testing.

### 1. Git dependency crash in Modal

Problem:
- `train.py` startup assumed `.git/` existed
- the Modal image intentionally excludes `.git/`
- training crashed before it even started

Resolution:
- made Git metadata collection non-fatal
- only write `gitdiff.patch` when Git is actually available

### 2. Credential leak risk from `.netrc`

Problem:
- an earlier version copied the local `.netrc` into the image
- that would have baked secrets into an image layer

Resolution:
- removed `.netrc` handling entirely
- remote W&B auth now only uses forwarded `WANDB_API_KEY`

### 3. Unsupported / outdated image design

Problem:
- the original plan used an old CUDA base image plus Miniconda
- the old image path was outdated
- Miniconda introduced additional friction

Resolution:
- moved to `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`
- dropped Miniconda entirely

### 4. Modal Python-version mismatch

Problem:
- Modal's current runner requires Python 3.10+
- Isaac Gym Preview 4 still needs Python 3.8
- a pure 3.8 image let the build succeed but the Modal runner crashed before function execution

Resolution:
- run Modal itself on standalone Python 3.11
- run training subprocesses on `/usr/bin/python3` (Python 3.8)

### 5. Remote config path bug

Problem:
- `modal_train.py` is mounted at `/root/modal_train.py`
- the project tree lives at `/root/project`
- a path derived from `__file__` broke inside the container

Resolution:
- config-path resolution now prefers `/root/project/...` when present and falls back safely to the local repo path

### 6. Isaac Gym `np.float` incompatibility

Problem:
- Isaac Gym's `torch_utils.py` still used `np.float`
- modern NumPy removes that alias
- stage 1 initially failed during import with:
  - `AttributeError: module 'numpy' has no attribute 'float'`

Resolution:
- patch Isaac Gym during the image build:
  - `dtype=np.float -> dtype=float`

### 7. W&B behavior in non-interactive environments

Problem:
- the first W&B integration could require auth even for smoke runs

Resolution:
- default unauthenticated runs to offline mode
- support explicit `WANDB_MODE`
- only go online automatically when `WANDB_API_KEY` is present

### 8. Incorrect / weak cache and checkpoint assumptions

Problem:
- the earlier cache check was too weak
- checkpoint handling did not fully reflect the actual stage-specific filenames used by training

Resolution:
- added full cache-set validation
- added centralized checkpoint-path helpers and tests

## Files Changed

Primary runtime files:
- `modal_train.py`
- `train.py`
- `hora/algo/ppo/ppo.py`
- `hora/algo/padapt/padapt.py`
- `hora/utils/misc.py`
- `hora/utils/wandb_utils.py`
- `hora/utils/checkpoint_utils.py`
- `hora/utils/graphics.py`
- `hora/tasks/base/vec_task.py`
- `requirements.txt`

Test/dev files:
- `requirements-dev.txt`
- `pytest.ini`
- `tests/test_modal_train.py`
- `tests/test_training_metadata.py`
- `tests/test_vec_task.py`
- `tests/test_wandb_utils.py`

## Current Status

The Modal integration is now in a usable state:
- local tests cover the harness and the critical supporting fixes
- live Modal smoke tests passed for cache setup, stage 1, and stage 2
- the remaining operational choice is whether remote runs should stay offline in W&B or receive `WANDB_API_KEY` for online sync
