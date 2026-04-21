

Commands Ran:

```bash
conda activate hora2
export WANDB_API_KEY=your_key_here

# baseline
modal run --detach modal_train.py --run-name baseline04201135 --runtime-profile a100_compat --stage 1
modal run --detach modal_train.py --run-name baseline04201135 --runtime-profile a100_compat --stage 2

# tactile
modal run --detach modal_train.py --run-name tactile04201119
--runtime-profile a100_compat --stage 1 --overrides "task.env.hora.useTactileObs=True"
modal run --detach modal_train.py --run-name tactile04201119 --runtime-profile a100_compat --stage 2 --overrides "task.env.hora.useTactileHist=True task.env.hora.useTactileObs=True"

