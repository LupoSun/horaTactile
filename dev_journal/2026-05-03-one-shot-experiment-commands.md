# One-Shot Experiment Commands

## Goal

Launch the full comparison set as durable one-shot Modal pipelines:

```text
Stage 1 -> Stage 2 -> Stage 2 BTG mean eval
```

These commands use the cloud-side pipeline path in `modal_train.py::main`, so `--stage both`
and `--auto-eval` should continue inside Modal after the local terminal closes.

## Shared Settings

```text
runtime_profile: a100_compat
pointcloud_points: 200
stage1 steps: 1.5B
stage2 steps: 500M
eval seeds: 3
```

The run names below use a `0503_1p5b` suffix to avoid colliding with older outputs.

## Smoke Test
```bash
modal run --detach modal_train.py::main \
--run-name fullbaseline_0504_smoke \
--runtime-profile a100_compat \
--stage both \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 1 \
--stage1-overrides "train.ppo.max_agent_steps=10000000" \
--stage2-overrides "train.ppo.max_agent_steps=10000000"
```

## Full Baseline

Shape-aware PointNet baseline, no direct tactile actor observations and no tactile adaptation
history.

```bash
modal run --detach modal_train.py::main \
--run-name fullbaseline_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=1500000000" \
--stage2-overrides "train.ppo.max_agent_steps=500000000"
```

## Full Tactile

PointNet plus direct tactile actor observations in Stage 1 and Stage 2, and tactile history
for Stage 2 adaptation.

```bash
modal run --detach modal_train.py::main \
--run-name fulltactile_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=1500000000" \
--stage2-overrides "train.ppo.max_agent_steps=500000000"
```

## PPO Asymmetric Critic

Tactile policy with privileged/shape information kept out of the actor and used by the critic.

```bash
modal run --detach modal_train.py::main \
--run-name ppo_asym_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_critic \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=1500000000" \
--stage2-overrides "train.ppo.max_agent_steps=500000000"
```

## PPO Recurrent

Tactile policy with a GRU encoder over the existing three-frame flattened observation window.

```bash
modal run --detach modal_train.py::main \
--run-name ppo_recurrent_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_recurrent \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=1500000000" \
--stage2-overrides "train.ppo.max_agent_steps=500000000"
```

## PPO Contact Gated

Tactile policy with contact-event mixture-of-experts action heads.

```bash
modal run --detach modal_train.py::main \
--run-name ppo_contact_gated_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_contact_gated \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=1500000000" \
--stage2-overrides "train.ppo.max_agent_steps=500000000"
```

## PPO Contact Reset

Tactile policy with contact-transition-reset recurrent encoding.

```bash
modal run --detach modal_train.py::main \
--run-name ppo_contact_reset_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_contact_reset \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=1500000000" \
--stage2-overrides "train.ppo.max_agent_steps=500000000"
```

## PPO Contact Aux

Tactile policy with a contact-transition auxiliary prediction loss on the actor trunk.

```bash
modal run --detach modal_train.py::main \
--run-name ppo_contact_aux_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_contact_aux \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=1500000000" \
--stage2-overrides "train.ppo.max_agent_steps=500000000"
```

## Notes

- Use `--auto-eval`, not `--autoeval`.
- Stage-specific overrides intentionally avoid sharing `train.ppo.max_agent_steps` across both
  stages.
- If reusing an old run name, delete or move the old output directory first; the training
  helpers check for existing checkpoints and refuse to overwrite.
