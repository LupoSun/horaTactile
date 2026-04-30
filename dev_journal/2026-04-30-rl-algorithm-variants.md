# RL Algorithm Variants

## Goal

Add a small set of Stage 1 RL experiment variants for comparing against the current PPO
oracle-policy baseline. Stage 2 remains the supervised adaptation/distillation stage.

## Variants Added

`modal_train.py::main` now accepts `--rl-variant`:

```text
ppo
ppo_recurrent
ppo_asym_critic
ppo_tuned
td3
```

`ppo` is the existing baseline.

`ppo_recurrent` keeps PPO but enables a GRU encoder over the existing flattened
three-frame actor observation window:

```text
train.ppo.recurrent_obs=True
train.ppo.recurrent_obs_seq_len=3
train.ppo.recurrent_hidden_size=128
```

This is a lightweight recurrent actor/critic comparison. It is not full hidden-state
PPO across rollout boundaries.

`ppo_asym_critic` keeps PPO but gives privileged extrinsic information only to the critic:

```text
train.ppo.asymmetric_critic=True
train.ppo.actor_use_privileged_info=False
```

The actor receives deployable observations only. The critic receives observation plus the
encoded privileged vector.

`ppo_tuned` keeps the model architecture fixed and changes the KL/entropy/batch schedule:

```text
train.ppo.kl_threshold=0.01
train.ppo.entropy_coef=0.001
train.ppo.horizon_length=16
train.ppo.minibatch_size=32768
train.ppo.mini_epochs=4
```

`td3` adds a first off-policy baseline:

```text
train.algo=TD3
train.ppo.td3_batch_size=32768
train.ppo.td3_learning_starts=80000
train.ppo.td3_replay_size=100000
```

TD3 is currently Stage 1-only and saves an actor/critic checkpoint format that is not
compatible with Stage 2 `ProprioAdapt`.

# Commands Ran

Recurrent PPO
```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_recurrent \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_recurrent \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"

modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_recurrent \
--runtime-profile a100_compat \
--stage 2 \
--tactile \
--rl-variant ppo_recurrent \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=200000000" \
--auto-eval \
--auto-eval-num-seeds 1
```

Asymmetric PPO
```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_asym \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_asym_critic \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"

modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_asym \
--runtime-profile a100_compat \
--stage 2 \
--tactile \
--rl-variant ppo_asym_critic \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=200000000" \
--auto-eval \
--auto-eval-num-seeds 1
```

Tuned PPO
```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_tuned \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_tuned \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"

modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_tuned \
--runtime-profile a100_compat \
--stage 2 \
--tactile \
--rl-variant ppo_tuned \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=200000000" \
--auto-eval \
--auto-eval-num-seeds 1
```

<!-- TD3
```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_td3_s1 \
--runtime-profile a100_compat \
--stage 1 \
--rl-variant td3 \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000" -->
