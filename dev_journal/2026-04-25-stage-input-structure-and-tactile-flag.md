# Stage Input Structure and Tactile Flag Semantics

## Why This Exists

This note is the exact tensor-level contract for the current shape-aware cylinder setup.
It clarifies what Stage 1 and Stage 2 receive with and without the Modal `--tactile`
flag.

## Important Flag Semantics

In `modal_train.py`, `--tactile` is a Stage 2 convenience flag. It does not add tactile
observations to Stage 1.

When `--stage both --tactile` is used:

- Stage 1 still trains the oracle with proprioception plus privileged object information
- Stage 1 actor observation remains 96D
- Stage 2 uses tactile only in the adaptation history
- Stage 2 actor observation remains Stage 1-compatible at 96D

Concretely, `--tactile` appends these Stage 2 overrides unless the user already provided
them:

```text
task.env.hora.useTactileObs=False
task.env.hora.useTactileHist=True
```

This choice is deliberate. Direct tactile observations would change the actor observation
shape and would require Stage 1 to have been trained with the same direct tactile
observation layout.

## Stage 1 Oracle Input

Stage 1 does not receive tactile in the current recommended setup, regardless of whether
the Modal command uses `--tactile`.

### Privileged Encoder Input

The privileged encoder receives:

```text
priv_info: 17D
point_cloud: N x 3, default N = 1024
```

Supported Modal point counts are currently `100`, `200`, `300`, `500`, and `1024`.

The 17D `priv_info` vector is:

```text
0:3    object position
3:4    object scale
4:5    object mass
5:6    object friction
6:9    object center of mass
9:13   object orientation quaternion
13:16  object angular velocity
16:17  restitution slot, currently zero unless randomized elsewhere
```

The point cloud is the object's canonical point cloud rotated by the current object
quaternion before PointNet. Translation is not applied; object position remains in
`priv_info`.

The encoder computes:

```text
z_phys  = MLP_17_to_8(priv_info)
z_shape = PointNet_Nx3_to_32(point_cloud)
z_t     = concat(z_phys, z_shape) = 40D
```

`z_t` is passed through `tanh` before concatenation with the actor observation.

### Control Policy Input

The Stage 1 actor receives:

```text
obs: 96D
tanh(z_t): 40D
total actor input: 136D
```

The 96D `obs` is a 3-frame window:

```text
3 x 32D
each 32D frame = 16 normalized joint positions + 16 current PD targets
```

The actor outputs:

```text
action: 16D
```

The action is applied as a PD target increment:

```text
cur_targets = prev_targets + action / 24
```

## Stage 2 Without `--tactile`

Without `--tactile`, Stage 2 is proprioception-only on the student side.

### Student Adaptation Input

The adaptation module receives:

```text
proprio_hist: 30 x 32
```

Each history step is:

```text
32D = 16 normalized joint positions + 16 current PD targets
```

The adaptation module outputs:

```text
z_hat: 40D
tanh(z_hat): 40D
```

### Control Policy Input

The Stage 2 actor receives:

```text
obs: 96D
tanh(z_hat): 40D
total actor input: 136D
```

The 96D `obs` has the same structure as Stage 1:

```text
3 x 32D
each 32D frame = 16 normalized joint positions + 16 current PD targets
```

### Teacher-Only Training Inputs

During Stage 2 training only, the teacher path receives:

```text
priv_info: 17D
point_cloud: N x 3, default N = 1024
```

These are used to compute oracle `z_t` and oracle action `a_t`. They are not student
inputs.

The Stage 2 losses are:

```text
L_z = ||z_hat - z_t||^2
L_a = ||a_hat - a_t||^2
L   = adapt_latent_loss_coef * L_z + adapt_action_loss_coef * L_a
```

Both coefficients currently default to `1.0`.

## Stage 2 With `--tactile`

With `--tactile`, Stage 2 adds tactile only to the adaptation history.

### Student Adaptation Input

The adaptation module receives:

```text
proprio_hist: 30 x 44
```

Each history step is:

```text
32D proprio/action history
12D tactile force history
```

The 12D tactile vector is:

```text
4 fingertips x 3 xyz net contact force
```

The tactile force is scaled by `/10` and clipped to `[-1, 1]`.

The adaptation output remains:

```text
z_hat: 40D
tanh(z_hat): 40D
```

### Control Policy Input

The Stage 2 actor input is unchanged by `--tactile`:

```text
obs: 96D
tanh(z_hat): 40D
total actor input: 136D
```

This is the key compatibility point: tactile history affects the predicted extrinsic
vector, not the actor observation dimensionality.

### Teacher-Only Training Inputs

The teacher-only training path is the same as the no-tactile case:

```text
priv_info: 17D
point_cloud: N x 3, default N = 1024
```

These compute `z_t` and `a_t` for the Stage 2 imitation losses.

## Stage 2 Eval and Deployment

Stage 2 eval/deployment uses only:

```text
obs
proprio_hist
```

It does not use:

```text
priv_info
point_cloud
```

For shape-aware Stage 2 eval, the model still needs `train.ppo.use_shape_priv_info=True`
so the checkpoint architecture has a 40D extrinsic path. The environment can set
`task.env.hora.useShapePrivInfo=False` so it does not load or generate point cloud
sidecars during eval.

# Experiments Ran

```bash
conda activate hora2
export WANDB_API_KEY=your_key_here

modal run --detach modal_train.py::main \
--run-name mixed_pointnet_tactile_04261105 \
--runtime-profile a100_compat \
--stage both \
--tactile \
--pointcloud-points 200 \
--auto-eval \
--auto-eval-num-seeds 1 \
--overrides "train.ppo.max_agent_steps=1500000000"
```
