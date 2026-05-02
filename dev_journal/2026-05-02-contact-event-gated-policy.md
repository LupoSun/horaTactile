# Contact-Event-Gated Policy Variant

## Motivation

Finger-gaiting is a hybrid control problem: the continuous joint controller is interrupted
by discrete contact make/break events. The current PPO and Stage 2 adaptation policies run a
fixed-frequency MLP over observation history, so the architecture has no explicit way to
change behavior when the tactile contact regime changes.

This experiment tests a small, compatible version of Angle 3: a tactile-gated mixture actor.
The policy keeps the same PPO objective, observation API, privileged-info path, Stage 2
distillation loop, and eval sweep infrastructure, but replaces the single actor action head
with a mixture of contact-mode action experts.

## Implementation

New RL variant:

```text
ppo_contact_gated
```

Overrides:

```text
train.ppo.contact_event_gating=True
train.ppo.contact_num_modes=4
train.ppo.contact_gate_hidden_size=32
train.ppo.contact_tactile_dim=12
train.ppo.contact_history_len=3
```

The actor trunk is unchanged. After the actor MLP, four linear action experts each propose an
action mean. A small gate reads tactile contact-event features from the tactile block at the
end of the observation vector:

```text
current tactile magnitude
absolute tactile transition from previous frame
mean tactile magnitude across the recent history
```

The final actor mean is the soft mixture of the expert means. The critic path, privileged
encoder, shape PointNet, recurrent option, and Stage 2 `ProprioAdapt` latent supervision are
unchanged.

## Compatibility Notes

- Requires `--tactile` for the contact gate to receive meaningful tactile observations.
- Uses the existing flattened tactile observation block, so there is no env API change.
- Stage 1 and Stage 2 must both use `--rl-variant ppo_contact_gated`, because the checkpoint
  contains `contact_gate` and `mu_experts` parameters instead of the default single `mu` head.
- Stage 2 eval also uses `--rl-variant ppo_contact_gated`. The auto eval manifest now preserves
  architecture overrides, so the generated eval subprocess should rebuild the same model.
- This is a soft mixture-of-experts rather than a hard options policy. It is the lower-risk
  first step toward contact-regime options because it avoids adding rollout-level option state
  or reset bookkeeping.

## Hypothesis

If tactile contact transitions are clean enough, the gate should specialize experts around
contact regimes such as low/no contact, contact make, contact break, and sustained contact.
That could improve finger-gaiting because the actor can choose different local action maps
around discrete contact changes without needing the shared MLP trunk to represent all regimes
with one final linear head.

## Commands

Stage 1:

```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_contact_gated \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_contact_gated \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"
```

Stage 2:

```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_contact_gated \
--runtime-profile a100_compat \
--stage 2 \
--tactile \
--rl-variant ppo_contact_gated \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=200000000"
```

Stage 2 BTG mean eval:

```bash
modal run --detach modal_train.py::stage2_eval \
--run-name mixed_pointnet_tactile_ppo_contact_gated \
--runtime-profile a100_compat \
--tactile \
--rl-variant ppo_contact_gated \
--pointcloud-points 200 \
--num-seeds 3
```

Short smoke-test version:

```bash
modal run --detach modal_train.py::main \
--run-name smoke_tactile_ppo_contact_gated \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_contact_gated \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=2000000"
```

## Success Criteria

- Stage 1 checkpoint loads into Stage 2 without state-dict mismatch.
- Stage 2 eval loads the checkpoint without missing or unexpected `contact_gate` /
  `mu_experts` keys.
- Reward is competitive with `ppo_tuned` or tactile baseline on BTG mean-object eval.
- If reward improves, add logging for contact gate entropy and per-mode usage to verify
  whether the modes specialize around contact events.

# Theory

The contact-event gated actor is a small mixture-of-experts policy head that lets the actor
choose different action mappings depending on what the fingertips are currently feeling.

The normal actor is:

```text
observation -> shared MLP -> one action head -> action mean
```

The contact-gated actor is:

```text
observation -> shared MLP -> several action heads
tactile contact-event features -> gate weights
final action mean = weighted mix of the action heads
```

Instead of one final linear layer producing the 16 Allegro joint action means, there are four
expert heads. Each expert proposes its own action mean. A small gating network looks at recent
tactile/contact signals and decides how much to use each expert.

The gate currently reads:

```text
current tactile magnitude
absolute tactile change from the previous frame
average tactile magnitude across recent frames
```

This is intended to expose rough contact regimes such as:

```text
no/low contact
new contact
lost contact
sustained contact
```

The regimes are not hard-coded. The gate and experts are learned end to end by PPO. For
example, the learned gate could put most weight on one expert during stable contact, another
expert when a fingertip makes contact, and another when contact breaks.

The reason this may help is that finger-gaiting is not just smooth continuous control. The
right local action map can change sharply when a fingertip touches or releases the object.
A single MLP head must represent all regimes with one final action map. The gated actor gives
the final action map a contact-conditioned structure while preserving the same PPO and Stage 2
training loops.

This is not yet a full options framework. It does not keep a discrete option state across time,
hard-reset recurrent state, or force experts to correspond to named modes. It is the lowest-risk
first step toward exploiting contact events architecturally without rewriting rollout storage,
environment stepping, or PPO.