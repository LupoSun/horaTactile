# Hierarchical Contact RL Experiment

## Goal

Push the non-asymmetric contact-gated policy beyond a soft mixture-of-experts head into a
true hierarchical contact policy.

The current baseline for this line is:

```text
ppo_contact_gated
```

That variant already gives the actor multiple contact-conditioned action heads, but it is
still a flat policy in an important sense: every timestep recomputes a soft mixture directly
from tactile features, and there is no persistent option state, no termination decision, and no
temporal abstraction.

The new experiment should ask whether contact events are not only useful as instantaneous gate
features, but also useful as natural boundaries between short-lived manipulation skills.

## Starting Point

`ppo_contact_gated` currently does:

```text
obs -> shared actor trunk -> K action experts
tactile contact-event features -> soft gate weights
action mean = weighted sum of expert means
```

The gate reads recent tactile force features:

```text
current tactile magnitude
absolute tactile transition
recent mean tactile magnitude
```

This is a good first contact-mode inductive bias, but it has three limitations:

- The gate can jitter every control step.
- Experts are blended, so no expert is forced to become a coherent temporally extended skill.
- There is no explicit notion of "stay in this local contact strategy until a contact event
  makes switching useful."

Hierarchical RL is the natural next step if we want the policy to discover contact-regime
skills instead of only contact-regime action interpolation.

## Design Choices

There are several plausible hierarchy designs.

### 1. Hard Contact Options With Learned Termination

Keep the current actor trunk and expert action heads, but replace the soft mixture with a
persistent discrete option:

```text
high-level option o_t in {1..K}
low-level action head mu_{o_t}(h_t)
termination beta_t decides whether to resample option
```

The high-level policy samples a new option only when:

```text
previous option terminates
or a strong tactile contact event occurs
or a max dwell length is reached
```

This is closest to the current contact-gated actor and is the recommended first hierarchical
experiment.

Pros:
- Clear extension of `ppo_contact_gated`.
- Experts become temporally coherent skills.
- Contact transitions become option-boundary candidates.
- Can be trained with PPO plus a small option log-prob term.

Cons:
- Requires rollout storage for option ids, option log-probs, termination probabilities, and
  dwell counters.
- Stage 2 inference needs persistent option state, so eval/deployment must carry option state
  across steps.

### 2. Soft Options With Dwell Regularization

Keep soft mixture weights, but add a temporal regularizer:

```text
L_switch = ||w_t - w_{t-1}||^2
```

and reduce the penalty around contact events:

```text
L_switch *= 1 - contact_event_score
```

This is a gentle bridge from MoE to hierarchy.

Pros:
- Almost no rollout API change.
- Low implementation risk.
- Good diagnostic before hard options.

Cons:
- Still not truly hierarchical.
- Experts can remain blended and ambiguous.
- Less publishable if the goal is explicitly hierarchical contact control.

### 3. Option-Critic Contact Policy

Implement an option-critic-style actor:

```text
intra-option policies pi(a | s, o)
termination policies beta_o(s)
high-level policy pi_O(o | s)
```

Options are trained end-to-end with PPO-style losses plus termination gradients.

Pros:
- Clean hierarchical RL framing.
- Learned termination can discover non-obvious contact boundaries.

Cons:
- More algorithmic surface area.
- More ways to destabilize PPO.
- Harder to keep compatible with Stage 2 distillation and current eval tools.

### 4. Pretrain Contact Experts, Then Train a Manager

First train `ppo_contact_gated`, then freeze or initialize the expert heads from it. Train only
a high-level manager that selects one expert at a time.

Pros:
- Uses existing trained contact-gated models.
- Manager training has a strong low-level initialization.

Cons:
- Requires checkpoint surgery and careful freezing/unfreezing.
- If the soft experts did not specialize, the manager starts from weak skills.

### 5. Contact Phase Prediction + Hierarchical Conditioning

Train an auxiliary contact-phase model:

```text
low/no contact
contact make
sustained contact
contact break
```

Then condition an option policy on the inferred phase.

Pros:
- Interpretable hierarchy.
- Easy to log and analyze.

Cons:
- Hand-designed phase bins may be too crude.
- Less end-to-end than a learned option policy.

## Recommended Experiment: `ppo_contact_options`

Implement hard contact options with learned termination, without asymmetric critic.

This deliberately stays on the same research branch as `ppo_contact_gated`:

```text
shape-aware oracle
full tactile actor observations
tactile Stage 2 history
no asymmetric critic
K = 4 contact options
```

The new actor becomes:

```text
obs_t -> shared actor trunk h_t
h_t -> K low-level action heads
contact features + h_t -> high-level option logits
contact features + h_t + current option -> termination probability
selected option o_t -> action mean mu_{o_t}(h_t)
```

The main difference from `ppo_contact_gated` is persistence:

```text
o_t = o_{t-1} unless termination or forced contact-event switch occurs
```

So the policy learns short-lived manipulation skills rather than recomputing a fresh mixture
every frame.

## Option State

Each environment tracks:

```text
current_option: int64, shape [num_envs]
option_dwell: int64, shape [num_envs]
previous_option_logprob: float, shape [num_envs]
previous_termination_logprob: float, shape [num_envs]
```

On environment reset:

```text
current_option = sample pi_O(o | obs)
option_dwell = 0
```

At each step:

```text
contact_event = event_score(obs_t) > threshold
terminate = sample beta(o_{t-1}, obs_t)
force_switch = option_dwell >= max_option_dwell

if terminate or force_switch or contact_event_forced_boundary:
    current_option = sample pi_O(o | obs_t)
    option_dwell = 0
else:
    current_option = previous_option
    option_dwell += 1
```

There are two variants worth testing:

```text
contact_boundary_mode=soft
```

Contact events are input features to the termination policy, but do not force switching.

```text
contact_boundary_mode=forced
```

Large contact events force a termination decision. The manager may still resample the same
option, but the option boundary is explicit.

The first implementation should start with `soft`; `forced` can be a follow-up ablation.

## Loss Terms

Use the normal PPO action loss for the selected low-level action distribution.

Add option-policy PPO loss only on timesteps where an option was newly sampled:

```text
L_option = clipped_ppo_loss(option_logprob, old_option_logprob, advantage)
```

Add termination regularization:

```text
L_term_sparsity = mean(beta_t)
```

This discourages terminating every frame.

Add dwell target regularization:

```text
L_dwell = max(0, min_dwell - option_dwell_before_termination)
```

This is optional and should start disabled unless the option policy collapses into single-step
switching.

Add entropy terms:

```text
H_action
H_option
H_termination
```

Use separate coefficients so action entropy and option entropy can be tuned independently.

Initial coefficients:

```text
contact_option_entropy_coef=0.002
contact_termination_entropy_coef=0.001
contact_termination_sparsity_coef=0.01
contact_min_dwell_loss_coef=0.0
```

## Config Proposal

New RL variant:

```text
ppo_contact_options
```

Suggested overrides:

```text
train.ppo.contact_options=True
train.ppo.contact_num_modes=4
train.ppo.contact_tactile_dim=12
train.ppo.contact_history_len=3
train.ppo.contact_gate_hidden_size=32
train.ppo.contact_gate_event_features=True
train.ppo.contact_gate_threshold=0.05
train.ppo.contact_option_max_dwell=12
train.ppo.contact_option_min_dwell=2
train.ppo.contact_option_boundary_mode=soft
train.ppo.contact_option_entropy_coef=0.002
train.ppo.contact_termination_entropy_coef=0.001
train.ppo.contact_termination_sparsity_coef=0.01
train.ppo.contact_min_dwell_loss_coef=0.0
```

This should not enable:

```text
train.ppo.asymmetric_critic=True
```

The point is to isolate hierarchy on top of the current non-asymmetric contact-gated idea.

## Implementation Status

Implemented on 2026-05-12 as:

```text
ppo_contact_options
```

Code paths added:

- `ContactOptionController` in `hora/algo/models/models.py`
- hard selected option action heads alongside the existing contact-gated MoE heads
- option-state storage in `hora/algo/ppo/experience.py`
- option policy / termination PPO losses in `hora/algo/ppo/ppo.py`
- persistent option state for Stage 1 train/test and Stage 2 train/test
- Modal preset wiring in `modal_train.py`
- focused unit coverage for the Modal preset and model-level contact-option path

The first implementation keeps option state outside the Isaac task. PPO and Stage 2 carry:

```text
current_option
option_dwell
reset_mask
force_switch_mask
```

and pass that state into the actor each step. This keeps the task API unchanged.

## Training Plan

Use the same object/training setup as the latest contact-gated branch:

```text
task.env.object.type=cylinder_default+custom_cylinder_2dcross+custom_cylinder_3dcross
task.env.object.sampleProb=[0.34,0.33,0.33]
pointcloud_points=200
full tactile actor observations
tactile Stage 2 history
shape-aware privileged oracle
```

Stage 1 trains the hierarchical oracle policy.

Stage 2 freezes the hierarchical actor and trains the adaptation module to predict the 40D
extrinsic latent. During Stage 2 rollout/eval, the actor still carries option state just like
Stage 1.

Proposed smoke run:

```bash
modal run --detach modal_train.py::main \
--run-name ppo_contact_options_0512_smoke \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_contact_options \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 1 \
--stage1-overrides "train.ppo.max_agent_steps=10000000" \
--stage2-overrides "train.ppo.max_agent_steps=10000000"
```

Primary A100 run:

```bash
modal run --detach modal_train.py::main \
--run-name ppo_contact_options_0512_750m \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_contact_options \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=750000000" \
--stage2-overrides "train.ppo.max_agent_steps=250000000"
```

Recommended ablation if the first run is stable:

```bash
modal run --detach modal_train.py::main \
--run-name ppo_contact_options_forced_0512_750m \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_contact_options \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=750000000 train.ppo.contact_option_boundary_mode=forced" \
--stage2-overrides "train.ppo.max_agent_steps=250000000 train.ppo.contact_option_boundary_mode=forced"
```

## Evaluation

Use the existing Stage 2 mean-BTG auto-eval:

```text
custom_btg1_mean ... custom_btg13_mean
3 seeds
rotate_reward as primary metric
```

Primary comparisons:

```text
fullbaseline_04291133
fulltactile_04291133
ppo_contact_gated_0503_1p5b
ppo_contact_options_0512_750m
```

If compute allows, compare against:

```text
ppo_contact_reset_0503_1p5b
ppo_contact_aux_0503_1p5b
```

The asymmetric contact-option variant should be treated as a companion ablation rather than the
main hierarchy result, because it confounds hierarchy with privileged critic changes.

## Companion Variant: `ppo_asym_contact_options`

Also implemented:

```text
ppo_asym_contact_options
```

This keeps the same hard contact-option actor, but adds:

```text
train.ppo.asymmetric_critic=True
train.ppo.actor_use_privileged_info=False
```

The motivation is the same as the earlier asymmetric PPO experiments: let the critic use
privileged shape/object information to reduce value-estimation noise while keeping the actor
deployable. In this variant, the deployable actor receives tactile/proprio observations and
maintains contact-option state, but it does not receive the privileged extrinsic vector.

Primary A100 run:

```bash
modal run --detach modal_train.py::main \
--run-name ppo_asym_contact_options_0512_750m \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_contact_options \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=750000000" \
--stage2-overrides "train.ppo.max_agent_steps=250000000"
```

This should be compared against:

```text
ppo_contact_options_0512_750m
ppo_asym_0503_1p5b
ppo_asym_contact_gated_v2_0504_1p5b
```

If `ppo_asym_contact_options` beats `ppo_contact_options`, the useful conclusion is narrower:
hierarchical contact options benefit from privileged critic stabilization. If both improve over
their non-hierarchical counterparts, the stronger conclusion is that temporal contact options
are useful independent of critic asymmetry.

## Pushing Further: Contact Options v2

The first contact-option implementation is intentionally conservative. If it does not improve
much over the corresponding flat/asymmetric baselines, the likely failure mode is that options
are not yet tied strongly enough to the contact-event structure. They may behave like a noisy
hard MoE head rather than temporally meaningful contact skills.

The next preset is:

```text
ppo_contact_options_v2
```

and the asymmetric companion is:

```text
ppo_asym_contact_options_v2
```

Changes from v1:

- `contact_option_boundary_mode=forced`
- `contact_option_max_dwell=8`
- `contact_option_min_dwell=3`
- `contact_option_entropy_coef=0.004`
- `contact_termination_sparsity_coef=0.02`
- `contact_min_dwell_loss_coef=0.01`
- `contact_option_balance_coef=0.02`

Interpretation:

- Forced boundaries make tactile contact transitions explicit option-boundary candidates
  instead of only latent inputs to the termination head.
- Shorter max dwell prevents one stale option from dominating long episodes.
- Min-dwell loss discourages degenerate one-step switching.
- Higher option entropy and balance loss fight option collapse.
- Higher termination sparsity prevents termination from becoming an every-frame default.

Primary A100 run:

```bash
modal run --detach modal_train.py::main \
--run-name ppo_contact_options_v2_0512_750m \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_contact_options_v2 \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=750000000" \
--stage2-overrides "train.ppo.max_agent_steps=250000000"
```

Asymmetric A100 companion:

```bash
modal run --detach modal_train.py::main \
--run-name ppo_asym_contact_options_v2_0512_750m \
--runtime-profile a100_compat \
--stage both \
--tactile \
--rl-variant ppo_asym_contact_options_v2 \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 3 \
--stage1-overrides "train.ppo.max_agent_steps=750000000" \
--stage2-overrides "train.ppo.max_agent_steps=250000000"
```

Success for v2 should not only be higher reward. It should also show healthier option
statistics:

```text
option usage less collapsed than v1
mean dwell > 1 and < max dwell
boundary rate visibly increases around contact events
termination rate not near 0 or 1
```

If v2 still does not help, the next hypothesis is that final-head options are too shallow. The
follow-up would be option-conditioned trunks or FiLM conditioning, where the selected option
modulates the actor hidden features before the action head instead of only selecting the final
linear head.

## Diagnostics To Log

Option usage:

```text
contact_options/option_0_usage
contact_options/option_1_usage
contact_options/option_2_usage
contact_options/option_3_usage
```

Temporal structure:

```text
contact_options/mean_dwell
contact_options/median_dwell
contact_options/termination_rate
contact_options/forced_boundary_rate
contact_options/same_option_resample_rate
```

Contact alignment:

```text
contact_options/contact_event_rate
contact_options/termination_given_event
contact_options/termination_without_event
contact_options/option_switch_given_make
contact_options/option_switch_given_break
```

Losses:

```text
losses/contact_option_policy_loss
losses/contact_termination_loss
losses/contact_termination_sparsity_loss
losses/contact_min_dwell_loss
```

The most important diagnostic is whether option switches align with contact make/break events
more often than chance while still producing nontrivial dwell times.

## Success Criteria

Minimum engineering success:

- Stage 1 trains without rollout storage or option-state shape errors.
- Stage 2 loads the Stage 1 checkpoint without missing option-policy keys.
- Stage 2 eval carries option state correctly across environment steps.
- Smoke eval completes on all 13 mean BTG objects.

Minimum research success:

- `ppo_contact_options` beats `ppo_contact_gated` on mean rotate reward averaged across
  BTG1-BTG13, or wins on at least 8 of 13 objects.
- Option dwell is meaningfully above one timestep.
- No single option takes more than 80% average usage.
- Termination rate increases around tactile contact events.

Stronger result:

- The hierarchical policy improves especially on mid-difficulty BTG objects where flat tactile
  and soft contact gating have room to improve, such as BTG3-BTG7 and BTG11-BTG13.

## Risks

The main risk is option collapse:

```text
one option dominates
or every option terminates every timestep
or the manager switches randomly without contact alignment
```

Countermeasures:

- Increase option entropy if one option dominates.
- Increase termination sparsity if termination happens every frame.
- Add mild load-balancing only if entropy is insufficient.
- Increase `contact_option_min_dwell` only after confirming single-step switching.

The second risk is PPO instability from adding discrete option log-probs. If this happens,
fallback to the softer bridge experiment:

```text
ppo_contact_gated + contact-aware dwell/switch regularization
```

That softer variant is less ambitious, but it can provide useful evidence before reattempting
hard options.

## Implementation Notes

Files likely to change:

- `configs/train/AllegroHandHora.yaml`
- `modal_train.py`
- `hora/algo/models/models.py`
- `hora/algo/ppo/experience.py`
- `hora/algo/ppo/ppo.py`
- `hora/algo/padapt/padapt.py`
- `tests/test_modal_train.py`
- `tests/test_wandb_utils.py`

The first implementation should avoid environment changes if possible. Option state can live
inside PPO/model inference wrappers, keyed by environment index and reset masks. That keeps the
task API stable and makes the experiment easier to compare with `ppo_contact_gated`.

## Takeaway

The next clean step after `ppo_contact_gated` is not just a bigger gate. It is giving contact
modes temporal identity.

`ppo_contact_options` tests the hypothesis that tactile contact events are natural option
boundaries for finger-gaiting. If it works, the project can claim a stronger idea than
"tactile features help a policy choose expert mixtures": contact events can structure a
hierarchical manipulation policy whose low-level skills persist across short contact regimes
and switch when the hand touches or releases the object.
