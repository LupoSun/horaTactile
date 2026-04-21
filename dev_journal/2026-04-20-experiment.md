

Commands Ran:

```bash
conda activate hora2
export WANDB_API_KEY=your_key_here

# baseline
modal run --detach modal_train.py::main --run-name baseline04211300 --runtime-profile a100_compat --stage 1
modal run --detach modal_train.py::main --run-name baseline04211300 --runtime-profile a100_compat --stage 2

# tactile
modal run --detach modal_train.py::main --run-name tactile04211301 --runtime-profile a100_compat --stage 1 --overrides "task.env.hora.useTactileObs=True"
modal run --detach modal_train.py::main --run-name tactile04211301 --runtime-profile a100_compat --stage 2 --overrides "task.env.hora.useTactileHist=True task.env.hora.useTactileObs=True"

# sweep
modal run --detach modal_train.py::eval_sweep \
--manifest configs/eval_sweeps/btg13_tactile04201119_5seeds.json \
--runtime-profile a100_compat
modal volume get hora-volume \
/outputs/eval_sweeps/btg13_tactile04201119_5seeds_20260421_175323 \
outputs/eval_sweeps/
