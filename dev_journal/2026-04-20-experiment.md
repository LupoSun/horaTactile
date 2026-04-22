## Commands Ran:

```bash
conda activate hora2
export WANDB_API_KEY=your_key_here

# baseline
modal run --detach modal_train.py::main --run-name baseline04211300 --runtime-profile a100_compat --stage 1
modal run --detach modal_train.py::main --run-name baseline04211300 --runtime-profile a100_compat --stage 2
modal run --detach modal_train.py::main \
--run-name baseline04201135 \
--runtime-profile a100_compat \
--stage 3 \
--overrides "train.ppo.max_agent_steps=100_000_000"


# tactile
modal run --detach modal_train.py::main --run-name tactile04211301 --runtime-profile a100_compat --stage 1 --overrides "task.env.hora.useTactileObs=True"
modal run --detach modal_train.py::main --run-name tactile04211301 --runtime-profile a100_compat --stage 2 --overrides "task.env.hora.useTactileHist=True task.env.hora.useTactileObs=True"
modal run --detach modal_train.py::main \
--run-name tactile04201119 \
--runtime-profile a100_compat \
--stage 3 \
--tactile \
--overrides "train.ppo.max_agent_steps=100_000_000"

# sweep
modal run --detach modal_train.py::eval_sweep \
--manifest configs/eval_sweeps/btg13_tactile04201119_5seeds.json \
--runtime-profile a100_compat
modal volume get hora-volume \
/outputs/eval_sweeps/btg13_tactile04201119_5seeds_20260421_175323 \
outputs/eval_sweeps/

# recording
python scripts/record_policy.py   --checkpoint outputs/AllegroHandHora/tactile04201119/stage2_nn/model_best.ckpt   --stage 2   --object-type custom_btg13_mean   --tactile-obs   --tactile-hist   --steps 400   --fps 20   --output outputs/recordings/tactile04201119__btg13_mean__stage2_best.mp4
```

## Notes and Fixes

### Modal command shape
Recent Modal CLI behavior requires explicitly selecting the local entrypoint when a file has more than one entrypoint. Training commands now use:

```bash
modal run --detach modal_train.py::main ...
```

instead of relying on:

```bash
modal run --detach modal_train.py ...
```
### Stage names and W&B
Stage 1 and Stage 2 must share the same `--run-name` because Stage 2 loads:

```text
outputs/AllegroHandHora/<run_name>/stage1_nn/best.pth
```

To avoid duplicate-looking W&B display names, W&B naming was separated from checkpoint/output naming. The shared output path remains unchanged, but W&B names now get stage suffixes:

```text
AllegroHandHora/<run_name>_s1
AllegroHandHora/<run_name>_s2
```

### Training length
The original HORA training budget was restored:

```yaml
train.ppo.max_agent_steps: 1_500_000_000
```

(we used 100M steps only, original was 1.5B)

This shared config applies to both Stage 1 PPO and Stage 2 ProprioAdapt unless a command-line override changes it.

### Tactile checkpoint compatibility
The tactile branch uses two independent flags:

```text
task.env.hora.useTactileObs
task.env.hora.useTactileHist
```

The intended modes are:

```text
vanilla Stage 2:  useTactileObs=False, useTactileHist=False
tactile Stage 1:  useTactileObs=True,  useTactileHist=False
tactile Stage 2:  useTactileObs=True,  useTactileHist=True
```

The old combined `task.env.hora.useTactile` override caused Hydra failures during evaluation because it is not present in the current structured config. The eval sweep runner was updated to emit only `useTactileObs` and `useTactileHist`.

## Thoughts
- Training steps need to be increased from 100M, but to 1.5B takes much longer time, maybe try to find middleground
- On object sizes:
  - seems only the smallest size is reasonable (mean)
- Since training is only on cylinders, it may not generalize well to our complicated BTG objects
  - the original paper tested on various objects, some of them are convex and more difficult, but not as difficult as our BTG
  - In smallest size our BTG is thin so very easy to be tilted
    - Maybe using thinner cylinders during traning? but that is not novel
- Looking at the videos, the performance on BTG object are very bad in general, either on vanilla or tactile
  - tactile has a lot of potential, but if training is only on cylinders, it may not learn how to use that signal well
  - Maybe we need to use a diversity of objects, like putting BTG objects in, like 2023 paper