# 2026-04-15 — H100 Tactile Stage-2 Experiment

## Summary
- Ran a full **Stage 2 tactile adaptation** experiment on Modal using the validated `h100_stable` runtime profile.
- Reused the existing upstream pretrained Stage 1 checkpoint `hora_v0.0.2/stage1_nn/best.pth` instead of retraining Stage 1 from scratch.
- The run completed and produced the expected Stage 2 artifacts:
  - `model_best.ckpt`
  - `model_last.ckpt`
  - `1.000m.ckpt`
- The offline Modal W&B run was downloaded locally and synced to the project W&B account.

## Why This Setup
- Tactile is only used in **Stage 2**, not Stage 1.
- We already had a known-good pretrained Stage 1 teacher checkpoint in:
  - `outputs/AllegroHandHora/hora_v0.0.2/stage1_nn/best.pth`
- That made it unnecessary to spend H100 time retraining the base policy.

## Run Setup
- **Runtime profile**: `h100_stable`
- **GPU**: `NVIDIA H100 80GB HBM3`
- **Algorithm**: `ProprioAdapt`
- **Task**: `AllegroHandHora`
- **Run name**: `h100_tactile_stage2_hora_v0_0_2_20260415_151825`
- **Stage 1 checkpoint used**: copied from upstream pretrained `hora_v0.0.2`
- **Tactile setting**: `task.env.hora.useTactile=True`
- **Object setting**: `task.env.object.type=cylinder_default`
- **Num envs**: `20000`
- **Max agent steps**: `100000000`

## Commands Used
### 1. Upload the pretrained Stage 1 checkpoint to the Modal volume
```bash
.modal-venv/bin/modal volume put hora-volume \
  outputs/AllegroHandHora/hora_v0.0.2/stage1_nn/best.pth \
  /outputs/AllegroHandHora/h100_tactile_stage2_hora_v0_0_2_20260415_151825/stage1_nn/best.pth
```

### 2. Launch Stage 2 tactile on H100
```bash
.modal-venv/bin/modal run modal_train.py \
  --run-name h100_tactile_stage2_hora_v0_0_2_20260415_151825 \
  --runtime-profile h100_stable \
  --stage 2 \
  --tactile
```

### 3. Download and sync the offline W&B run
```bash
.modal-venv/bin/modal volume get hora-volume \
  /wandb/wandb/offline-run-20260415_222145-fp8i8kmc \
  /tmp/h100_tactile_stage2_sync/

.venv/bin/wandb sync \
  /tmp/h100_tactile_stage2_sync/offline-run-20260415_222145-fp8i8kmc
```

## Saved Outputs
### Modal volume output directory
```text
/outputs/AllegroHandHora/h100_tactile_stage2_hora_v0_0_2_20260415_151825/
```

### Files confirmed on volume
- `stage1_nn/best.pth`
- `stage2_nn/model_best.ckpt`
- `stage2_nn/model_last.ckpt`
- `stage2_nn/1.000m.ckpt`
- `config_041522_nogit.yaml`

## W&B
- **Offline Modal run directory**: `offline-run-20260415_222145-fp8i8kmc`
- **Synced W&B run URL**:
  - `https://wandb.ai/tao_sun-university-of-california-berkeley/hora/runs/fp8i8kmc`

## Final Recorded Metrics
From the synced `wandb-summary.json`:
- `_step`: `100000000`
- `episode_rewards/step`: `78.90493462290796`
- `episode_lengths/step`: `235.75074534100952`
- `_runtime`: `668.532501314`

From the final training log:
- `Agent Steps: 0100M`
- `FPS: 149721.0`
- `Last FPS: 112336.5`
- `Current Best: 80.37`

## Notes / Observations
- The run correctly picked up the tactile-expanded history shape:
  - `RunningMeanStd: (30, 44)`
- This confirms the merged tactile path was actually active during Stage 2.
- The run used the robust fingertip-body resolution helper rather than the earlier brittle name lookup.
- Modal still launched the training container in W&B offline mode because `WANDB_API_KEY` was not exported into the shell before `modal run`.
  - That did **not** affect training correctness.
  - The run was synced successfully afterward from the downloaded offline artifact.

## Takeaway
- We now have a completed **H100 Stage 2 tactile adaptation run** using the upstream pretrained Stage 1 teacher.
- This is a much better baseline for comparison than the earlier smoke runs because it used the full default Stage 2 budget (`100M` agent steps).
- The main next step is to compare this tactile Stage 2 run against:
  - the pretrained non-tactile Stage 2 baseline (`hora_v0.0.2`)
  - any future tactile variants beyond simple fingertip contact-force concatenation
