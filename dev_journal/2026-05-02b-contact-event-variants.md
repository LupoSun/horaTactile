# Contact-Event Variants

## Goal

Build a family of variants around the idea that finger-gaiting is a hybrid control problem:
mostly continuous control, but with important discrete contact make/break events. The first
variant on 2026-05-02 used contact events to gate a mixture actor head. This entry sketches
more variants and starts the next most promising one.

## Variant Menu

### 1. Contact-Transition-Reset Recurrent Policy

Treat tactile contact transitions as soft reset events inside the recurrent observation
encoder. The policy still receives the usual flattened observation, but the encoder rebuilds
the observation into per-frame proprioceptive and tactile slices:

```text
[p0, p1, p2, t0, t1, t2] -> [(p0,t0), (p1,t1), (p2,t2)]
```

At each frame boundary, a small learned reset gate reads:

```text
current tactile magnitude
absolute tactile change from previous frame
```

and uses that to partially reset the recurrent hidden state before processing the new frame.
This is a soft, differentiable approximation of "reset memory on contact regime change."

This is the variant implemented first as:

```text
ppo_contact_reset
```

### 2. Contact-Event Gated Actor Head

Implemented in the previous entry as `ppo_contact_gated`. The shared actor trunk stays fixed,
but the action mean is a soft mixture over several expert action heads. The gate reads tactile
contact-event features and chooses which expert mixture to use.

This is a low-risk mixture-of-experts approximation to contact-mode options.

### 3. Contact Mode Embedding

Append a learned contact-mode embedding to the actor input. The mode could be a hard or soft
bin derived from fingertip contact occupancy and contact transitions:

```text
no contact
single-finger contact
multi-finger contact
new contact
lost contact
```

This is simpler than MoE. It asks whether explicit event labels help the same MLP discover
mode-conditioned behavior.

### 4. Contact-Transition Auxiliary Loss

Add an auxiliary head that predicts near-future contact transitions from the policy latent.
The reward and actor remain unchanged, but representation learning encourages the trunk to
encode make/break events.

This is attractive if contact events are detectable but the RL signal is too sparse/noisy to
force the policy to care about them.

### 5. Contact Option Dwell Regularization

Turn contact modes into soft options and add a regularizer that penalizes high-frequency
option switching except during tactile transition events. This is closer to hierarchical RL:
each option has a local action head, and contact transitions are treated as natural option
termination points.

This is more novel, but it requires rollout-level option-state bookkeeping and is riskier.

### 6. Finger-Set Contact Graph Policy

Represent active fingertip contacts as a small graph or set. Use a permutation-aware encoder
over active fingertips, then condition the actor on the active contact set. This aims to
generalize across which finger is in contact rather than only across raw force magnitudes.

This is promising if the policy needs finger-specific mode selection, but it needs more careful
tactile feature engineering.

## Started Variant: `ppo_contact_reset`

`ppo_contact_reset` enables recurrent observation encoding and replaces the normal fixed GRU
sequence encoder with a contact-transition-reset encoder. The reset is learned and soft:

```text
h_t_before = h_{t-1} * (1 - reset_gate(contact_t, abs(contact_t - contact_{t-1})))
h_t = GRUCell(frame_t, h_t_before)
```

The reset gate is scalar per environment and frame boundary. It can learn to preserve memory
during smooth contact and flush memory when a contact regime changes.

Why this is the most promising next step:

- It is closer to the discrete contact-event idea than a pure MoE head.
- It does not require environment changes or rollout storage changes.
- It fixes the frame structure for tactile recurrent observations by rebuilding
  `(proprio_frame, tactile_frame)` pairs before recurrence.
- It should remain compatible with Stage 2 adaptation and Stage 2 eval as long as all stages
  use the same `--rl-variant ppo_contact_reset`.

## Overrides

```text
train.ppo.recurrent_obs=True
train.ppo.recurrent_obs_seq_len=3
train.ppo.recurrent_hidden_size=128
train.ppo.contact_reset_recurrent=True
train.ppo.contact_tactile_dim=12
train.ppo.contact_gate_hidden_size=32
```

## Commands

Stage 1:

```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_contact_reset \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_contact_reset \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=750000000"
```

Stage 2:

```bash
modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_ppo_contact_reset \
--runtime-profile a100_compat \
--stage 2 \
--tactile \
--rl-variant ppo_contact_reset \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=200000000"
```

Stage 2 BTG mean eval:

```bash
modal run --detach modal_train.py::stage2_eval \
--run-name mixed_pointnet_tactile_ppo_contact_reset \
--runtime-profile a100_compat \
--tactile \
--rl-variant ppo_contact_reset \
--pointcloud-points 200 \
--num-seeds 3
```

Short smoke-test version:

```bash
modal run --detach modal_train.py::main \
--run-name smoke_tactile_ppo_contact_reset \
--runtime-profile a100_compat \
--stage 1 \
--tactile \
--rl-variant ppo_contact_reset \
--pointcloud-points 200 \
--overrides "train.ppo.max_agent_steps=2000000"
```

## Success Criteria

- Stage 1 trains without observation-shape errors.
- Stage 2 loads the Stage 1 checkpoint without state-dict mismatch.
- Stage 2 eval loads the Stage 2 checkpoint with the same recurrent reset architecture.
- Reward beats or matches `ppo_recurrent` and is competitive with `ppo_contact_gated`.
- If promising, log mean reset probability and reset probability around contact transitions.
