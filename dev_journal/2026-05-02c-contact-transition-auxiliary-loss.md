# Contact-Transition Auxiliary Loss

## Goal

Start Variant 4 from the contact-event menu: use tactile contact transitions as a
self-supervised auxiliary target. The policy and reward stay unchanged, but the actor trunk is
encouraged to encode whether fingertip contacts are about to make or break after the current
action.

## Theory

The previous contact-event variants changed the policy architecture directly:

```text
ppo_contact_gated: contact events gate action experts
ppo_contact_reset: contact events softly reset recurrent memory
```

`ppo_contact_aux` is different. It does not directly change the action computation. Instead,
it adds a representation-learning pressure on the actor latent.

During rollout, PPO stores:

```text
obs_t
action_t
obs_{t+1}
```

The next observation contains the recent tactile history. From `obs_{t+1}`, we derive a binary
target for each fingertip contact-force channel:

```text
contact_transition = contact_now XOR contact_previous
```

where contact is defined by a small tactile magnitude threshold. The actor trunk predicts this
next-step transition vector through an auxiliary head. PPO then optimizes:

```text
loss = ppo_loss + contact_transition_aux_coef * BCE(predicted_transition, transition_target)
```

This tests whether explicitly predicting contact make/break events helps the policy learn a
better tactile representation, even when the action head remains a standard MLP head.

## Why This Variant

This is useful if contact events are detectable but too weak or noisy in the reward signal to
shape the representation on their own. It is lower risk than full options because it does not
require option state, termination logic, or rollout bookkeeping beyond storing one auxiliary
target.

It is also a good diagnostic. If auxiliary prediction loss falls but reward does not improve,
then the network can detect contact transitions but the current control objective may not be
using them. If reward improves, contact-event representation is likely useful even before
building a more explicit options policy.

## Implementation

New RL variant:

```text
ppo_contact_aux
```

Overrides:

```text
train.ppo.contact_transition_aux_loss=True
train.ppo.contact_transition_aux_coef=0.05
train.ppo.contact_transition_aux_threshold=0.05
train.ppo.contact_tactile_dim=12
train.ppo.contact_history_len=3
```

The model adds `contact_transition_head` on top of the actor features. The PPO rollout buffer
stores `contact_transition_targets`, computed after each environment step from the next raw
observation. Done environments have the target zeroed to avoid training on reset artifacts.

Compatibility notes:

- Requires `--tactile`; without tactile observations the target becomes all zeros.
- Stage 1, Stage 2, and eval should all use `--rl-variant ppo_contact_aux` so checkpoint keys
  match.
- Stage 2 loads and preserves the auxiliary head, but its adaptation loss does not train it.
  This is intentional: the auxiliary head is a Stage 1 representation-shaping tool.
- The auto eval manifest preserves the variant overrides, so eval should rebuild the same
  architecture.

## Commands

Stage 1:

```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_contact_aux \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_contact_aux \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"
```

Stage 2:

```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_contact_aux \
--runtime-profile a100_compat \
--stage 2 \
--tactile \
--rl-variant ppo_contact_aux \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=200000000"
```

Stage 2 BTG mean eval:

```bash
modal run --detach modal_train.py::stage2_eval \
--run-name mixed_pointnet_tactile_ppo_contact_aux \
--runtime-profile a100_compat \
--tactile \
--rl-variant ppo_contact_aux \
--pointcloud-points 200 \
--num-seeds 3
```

Short smoke-test version:

```bash
modal run --detach modal_train.py::main \
--run-name smoke_tactile_ppo_contact_aux \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_contact_aux \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=2000000"
```

## Success Criteria

- Stage 1 trains without storage or shape errors.
- W&B logs `losses/contact_transition_aux_loss`.
- Stage 2 loads the Stage 1 checkpoint without missing `contact_transition_head` keys.
- Stage 2 eval loads the Stage 2 checkpoint with the same auxiliary-head architecture.
- Reward is competitive with `ppo_contact_gated` and `ppo_contact_reset`.
- If promising, add metrics for target positive rate, prediction accuracy, and per-finger
  transition precision/recall.
