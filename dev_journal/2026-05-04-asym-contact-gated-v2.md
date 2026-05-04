# Asymmetric Contact-Gated v2

## Goal

Push the strongest current idea: asymmetric PPO plus a contact-event-gated actor. The first
`asym + contact_gated` result suggests that the mixture actor can beat plain tactile on some
objects. This v2 tries to turn that into a more reliable gain by making the gate more explicitly
contact-event-aware and by discouraging expert collapse.

## Theory

The core hypothesis is that finger-gaiting is a hybrid control problem. Within a stable contact
regime, the policy should behave like smooth continuous control. Around contact make/break
events, the useful action map can change sharply. A single actor head has to represent all of
those regimes in one map.

The contact-gated actor gives the final action map a mode structure:

```text
deployable obs -> shared actor trunk -> expert action heads
tactile contact-event features -> soft gate weights
final action = weighted mixture of expert actions
```

The asymmetric critic keeps privileged shape/object information during PPO training, but the
actor remains deployable. The hope is that the critic lowers value-estimation noise while the
gate learns reusable contact regimes such as stable contact, new contact, lost contact, and
low/no contact.

## v2 Changes

This version keeps the same basic mixture-of-experts actor, but improves the gate:

- Explicit contact features: current contact, previous contact, force delta, absolute event
  magnitude, contact make, contact break, contact duration, active-contact count.
- Expert load balancing: a mild batch-level loss encourages the average gate usage to avoid
  collapsing into one expert.
- Gate switch penalty: a mild loss discourages high-frequency gate jitter by comparing current
  gate weights with the gate weights implied by the previous tactile frame.
- Diagnostics: log gate entropy, max gate probability, per-expert usage, active-contact rate,
  contact-event rate, make/break rates, balance loss, and switch loss.

This is still a soft options-style policy rather than a hard hierarchical controller. That is
intentional: we get the contact-mode inductive bias without adding option termination state or
changing rollout storage.

## Implementation

New named Modal/RL preset:

```text
ppo_asym_contact_gated_v2
```

The preset expands to:

```text
train.ppo.asymmetric_critic=True
train.ppo.actor_use_privileged_info=False
train.ppo.contact_event_gating=True
train.ppo.contact_num_modes=4
train.ppo.contact_gate_hidden_size=32
train.ppo.contact_tactile_dim=12
train.ppo.contact_history_len=3
train.ppo.contact_gate_event_features=True
train.ppo.contact_gate_threshold=0.05
train.ppo.contact_gate_balance_coef=0.01
train.ppo.contact_gate_switch_coef=0.005
```

Stage 1, Stage 2, and eval must all use this same preset. The v2 gate has a larger first layer
than the original contact-gated model, so its checkpoints are intentionally not shape-compatible
with older `contact_gated` runs.

## Command

Primary full run:

```bash
modal run --detach modal_train.py::main \
--run-name ppo_asym_contact_gated_v2_0504_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_contact_gated_v2 \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=500000000" \
--stage2-overrides "train.ppo.max_agent_steps=200000000"
```

Smoke test:

```bash
MODAL_CPU=8 modal run --detach modal_train.py::main \
--run-name ppo_asym_contact_gated_v2_0504_smoke \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_contact_gated_v2 \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 1 \
--stage1-overrides "train.ppo.max_agent_steps=10000000" \
--stage2-overrides "train.ppo.max_agent_steps=10000000"
```

## What To Watch

- `contact_gate/expert_*_usage`: experts should not collapse to one mode.
- `contact_gate/entropy`: should be neither fully uniform nor fully deterministic early.
- `contact_gate/contact_event_rate`: confirms the gate is seeing nontrivial tactile events.
- `losses/contact_gate_balance_loss`: should stay small but not dominate PPO.
- `losses/contact_gate_switch_loss`: should reduce jitter without freezing the gate.
- Eval reward by object: the target is more consistent gains over plain tactile, not only one
  lucky object.
