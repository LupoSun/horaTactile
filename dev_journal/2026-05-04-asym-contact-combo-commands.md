# Asymmetric PPO + Contact-Event Combo Commands

## Goal

Run the three contact-event variants combined with asymmetric-critic PPO:

```text
ppo_asym_critic + contact_gated
ppo_asym_critic + contact_reset
ppo_asym_critic + contact_aux
```

These are one-shot durable Modal pipelines:

```text
Stage 1 -> Stage 2 -> Stage 2 BTG mean eval
```

## Shared Settings

```text
runtime_profile: a100_compat
pointcloud_points: 200
stage1 steps: 1.5B
stage2 steps: 200M
eval seeds: 3
base RL variant: ppo_asym_critic
cpu request: 8 physical cores per Modal container
```

The contact-event flags are passed through shared `--overrides` so both training stages and
the generated auto-eval manifest use the same architecture.

## Asym + Contact Gated

Asymmetric critic plus contact-event mixture-of-experts actor heads.

Theory:

This variant treats contact regimes as latent modes of the manipulation problem. The actor
keeps a shared trunk, but replaces the single action-mean head with several expert heads.
A small tactile gate reads recent contact magnitudes and contact changes, then outputs soft
weights over the experts. The final action mean is the weighted mixture of those expert
actions.

The asymmetric critic provides privileged shape/object information during PPO training, while
the actor still has to act from deployable observations. In theory, the critic should reduce
gradient noise and make it easier for the contact gate to specialize: one expert can learn
stable-contact behavior, another can learn contact-make corrections, another can learn
contact-break recovery, without forcing a single final linear head to cover every regime.

```bash
MODAL_CPU=8 modal run --detach modal_train.py::main \
--run-name ppo_asym_contact_gated_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_critic \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--overrides "train.ppo.contact_event_gating=True train.ppo.contact_num_modes=4 train.ppo.contact_gate_hidden_size=32 train.ppo.contact_tactile_dim=12 train.ppo.contact_history_len=3" \
--stage1-overrides "train.ppo.max_agent_steps=500000000" \
--stage2-overrides "train.ppo.max_agent_steps=200000000"
```

## Asym + Contact Reset

Asymmetric critic plus contact-transition-reset recurrent encoding.

Theory:

This variant treats contact transitions as natural memory boundaries. A recurrent actor can
use tactile history to infer short-term object/finger state, but that memory can become stale
when a fingertip makes or breaks contact. The reset model adds a learned gate that reads the
current tactile contact signal and its change from the previous frame, then softly clears part
of the recurrent hidden state before processing the next observation.

The asymmetric critic is useful here because the reset gate is a more indirect architectural
bias than the mixture head. PPO still learns the actor from deployable observations, but the
privileged critic gives a cleaner value target while the recurrent encoder discovers when
history should be preserved versus flushed. The hypothesis is that finger-gaiting has episodes
of smooth continuous control separated by contact events, so the actor benefits from memory
that is stable within a contact regime and deliberately refreshed at regime boundaries.

```bash
MODAL_CPU=8 modal run --detach modal_train.py::main \
--run-name ppo_asym_contact_reset_0504_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_critic \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--overrides "train.ppo.recurrent_obs=True train.ppo.recurrent_obs_seq_len=3 train.ppo.recurrent_hidden_size=128 train.ppo.contact_reset_recurrent=True train.ppo.contact_tactile_dim=12 train.ppo.contact_gate_hidden_size=32" \
--stage1-overrides "train.ppo.max_agent_steps=500000000" \
--stage2-overrides "train.ppo.max_agent_steps=200000000"
```

## Asym + Contact Aux

Asymmetric critic plus the contact-transition auxiliary prediction loss.

Theory:

This variant does not directly change the action computation. Instead, it adds a
self-supervised contact-transition prediction task on top of the actor representation. The
target is whether each tactile contact channel changed state, roughly contact-now XOR
contact-previous, using a small tactile threshold. PPO optimizes the normal policy/value loss
plus a weighted binary cross-entropy loss for contact-transition prediction.

Combined with an asymmetric critic, this separates two kinds of supervision. The critic uses
privileged information to make policy optimization less noisy, while the auxiliary head forces
the deployable actor trunk to encode the contact make/break structure that may be weakly
expressed in reward. If auxiliary loss falls but reward does not improve, contact events are
detectable but not useful to the current control objective. If reward improves, it suggests
that contact-transition awareness is a useful representation bias even before adding explicit
options or hard mode switches.

```bash
MODAL_CPU=8 modal run --detach modal_train.py::main \
--run-name ppo_asym_contact_aux_0503_1p5b \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_critic \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--overrides "train.ppo.contact_transition_aux_loss=True train.ppo.contact_transition_aux_coef=0.05 train.ppo.contact_transition_aux_threshold=0.05 train.ppo.contact_tactile_dim=12 train.ppo.contact_history_len=3" \
--stage1-overrides "train.ppo.max_agent_steps=500000000" \
--stage2-overrides "train.ppo.max_agent_steps=200000000"
```

## Notes

- Use `--rl-variant ppo_asym_critic` for all three so the critic gets privileged/shape
  information while the actor remains deployable.
- The contact flags live in `--overrides`, because `modal_train.py` currently accepts one
  named `--rl-variant` preset at a time.
- `--tactile` is required for these contact-event variants to receive meaningful tactile
  observations.
- If a smoke test is desired first, replace the stage overrides with:

```text
--stage1-overrides "train.ppo.max_agent_steps=10000000"
--stage2-overrides "train.ppo.max_agent_steps=10000000"
```
