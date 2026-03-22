# Setup Journal

Development environment setup log for the HORA (In-Hand Object Rotation via Rapid Motor Adaptation) course project.

**Machine**: WSL2 on Windows laptop, NVIDIA RTX 3500 Ada Generation (sm_89), driver 581.60  
**Host OS**: Ubuntu 24.04 (Noble) under WSL2, kernel 6.6.87.2-microsoft-standard-WSL2

---

## 2025-03-21 — Initial setup and Isaac Gym install

### Python environment

The system Python is 3.12; Isaac Gym Preview 4 requires 3.8. We used `uv` (already installed) to manage a project-local venv:

```bash
uv python install 3.8
uv venv --python 3.8 .venv
source .venv/bin/activate
```

### Isaac Gym Preview 4.0

Downloaded from Google Drive via `gdown`, extracted into `isaacgym/` at the repo root, and installed editable:

```bash
gdown '1StaRl_hzYFYbJegQcyT7-yjgutc6C7F9' -O isaac4.tar.gz
tar -xzf isaac4.tar.gz
pip install -e isaacgym/python
pip install -r requirements.txt
```

Both `isaac4.tar.gz` and `isaacgym/` are git-ignored (see `.gitignore`).

### PyTorch version mismatch (critical)

The upstream docs pin PyTorch 1.10.1+cu102, which only supports up to sm_70.
Our RTX 3500 Ada is sm_89 — PyTorch was silently producing bad GPU kernels and Isaac Gym segfaulted.
**Fix**: upgraded to PyTorch 2.1.2 + CUDA 11.8:

```bash
uv pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

Also cleared the stale gymtorch JIT cache:

```bash
rm -rf ~/.cache/torch_extensions/py38_cu102
```

### WSL2 libcuda path

Isaac Gym's PhysX needs `libcuda.so`, which lives at `/usr/lib/wsl/lib` on WSL2.
Without it, PhysX falls back to CPU and headless GPU sims fail.
**Fix**: `scripts/isaac_wsl_env.sh` exports `LD_LIBRARY_PATH`.

### Vulkan viewer segfault

Isaac Gym's viewer uses Vulkan for rendering. On WSL2, `vulkaninfo` initially only showed
`llvmpipe` (CPU software rasterizer) — no GPU Vulkan device. The viewer segfaulted after
a few frames regardless of physics backend, asset, or environment count.

**Root cause**: Ubuntu 24.04's `mesa-vulkan-drivers` does not include the `dozen` (`dzn`)
driver, which translates Vulkan → DirectX 12 for WSL2 GPU access.

**Fix**: installed the [kisak-mesa PPA](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa),
which ships Mesa 26.x with the `dzn` driver pre-built:

```bash
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade
sudo apt install vulkan-tools mesa-vulkan-drivers
```

Then set `VK_ICD_FILENAMES` to point at the `dzn` ICD (also in `scripts/isaac_wsl_env.sh`):

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json
```

After this, `vulkaninfo --summary` shows the RTX 3500 Ada as a discrete GPU via the Dozen
driver, and the Isaac Gym viewer runs without crashing.

### Verified working

| Test | Result |
|------|--------|
| `import isaacgym; import torch` | OK |
| GPU PhysX + GPU pipeline (headless) | OK |
| `joint_monkey_wsl.py` with viewer (1 env, cartpole) | OK, clean exit |

### Session startup (every time)

```bash
cd /home/lupo/horaTactile
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
```

### Key references

- [IsaacGymEnvs#52](https://github.com/isaac-sim/IsaacGymEnvs/issues/52) — segfault discussion; driver alignment, Vulkan, tmux caveats.
- [wslg#1254](https://github.com/microsoft/wslg/issues/1254) — dozen driver for WSL2 Vulkan.
- [kisak-mesa PPA](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa) — pre-built Mesa with dzn for Ubuntu 24.04.

---

## 2025-03-22 — Joint monkey demo, grasp cache, training smoke

### Joint monkey (Isaac Gym viewer sanity check)

Use WSL env + venv, then the WSL-friendly script (1 env, cartpole by default):

```bash
cd /path/to/horaTactile
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
bash scripts/wsl_joint_monkey.sh
# or: python scripts/joint_monkey_wsl.py --sim_device cpu --asset_id 2 --num_envs 1
```

Upstream `isaacgym/python/examples/joint_monkey.py` uses **36 parallel envs**, which is hard on WSL; `scripts/joint_monkey_wsl.py` adds `--num_envs` (default 1).

### Grasp cache (required for HORA training)

From the repo root (see main `README.md`):

```bash
gdown '1xqmCDCiZjl2N7ndGsS_ZvnpViU7PH7a3' -O cache/data.zip
```

If `unzip` is not installed, extract with Python:

```bash
python -c "import zipfile; zipfile.ZipFile('cache/data.zip').extractall('cache/')"
```

This populates `cache/*_allegro_grasp_50k_s*.npy` for internal and public Allegro configs.

### HORA training — fixes applied in this repo

1. **Import order (Isaac Gym)**  
   `hora/tasks/allegro_hand_hora.py` and `allegro_hand_grasp.py` had `import torch` **before** `from isaacgym import gymtorch`, which breaks `gymdeps` (torch must not be loaded before Isaac Gym’s `gymtorch` path).  
   **Fix**: `gymtorch` / `gymapi` / `torch_utils` first, then `import torch`.

2. **`train.py`**  
   Import `hora.tasks.isaacgym_task_map` **before** `PPO` / `ProprioAdapt` so task modules load `gymtorch` before any `torch` import from the trainers.

3. **`isaacgym/python/isaacgym/torch_utils.py`**  
   Default arg `dtype=np.float` fails on modern NumPy (`np.float` removed). **Fix**: use `dtype=float`.

4. **Smoke run with fewer envs**  
   Default `numEnvs` is very large; for a laptop use e.g. `num_envs=512`. Then **`train.ppo.minibatch_size`** must divide **`horizon_length × num_envs`** (e.g. `8 × 512 = 4096` → set `train.ppo.minibatch_size=4096`).

Example **short** stage-1 smoke (same logic as `scripts/train_s1.sh`, capped steps):

```bash
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python train.py task=AllegroHandHora headless=True seed=0 \
  task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
  train.algo=PPO \
  task.env.object.type=cylinder_default \
  train.ppo.priv_info=True train.ppo.proprio_adapt=False \
  train.ppo.output_name=AllegroHandHora/smoke_train_s1 \
  num_envs=512 \
  train.ppo.max_agent_steps=65536 \
  train.ppo.minibatch_size=4096 \
  train.ppo.save_frequency=999999
```

**Result**: run completed with `max steps achieved` and improving reward prints.

Full training: use `scripts/train_s1.sh ${GPU_ID} ${SEED} ${RUN_NAME}` without the low `max_agent_steps` (default is very large).

### Official pretrained checkpoint

Downloaded and extracted the upstream pretrained release from the main README:

```bash
cd outputs/AllegroHandHora
gdown '17fr40KQcUyFXz4W1ejuLTzRqP-Qu9EPS' -O hora_v0.0.2.zip
python -c "import zipfile; zipfile.ZipFile('hora_v0.0.2.zip').extractall('hora_v0.0.2')"
```

The zip contained one extra nested `hora_v0.0.2/` directory, so we flattened it to match the README's expected layout:

```text
outputs/AllegroHandHora/hora_v0.0.2/
  stage1_nn/
  stage2_nn/
```

### Pretrained verification

1. **Viewer**  
   Stage-1 visualization launched successfully with:

```bash
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
bash scripts/vis_s1.sh hora_v0.0.2
```

2. **Headless evaluation**  
   A lighter eval run using the stage-1 checkpoint and `task.env.numEnvs=2048` showed steadily increasing reward, confirming the pretrained policy runs correctly on this machine.

3. **Stage-2 visualization**  
   The adaptation checkpoint also launched successfully with:

```bash
source scripts/isaac_wsl_env.sh
source .venv/bin/activate
bash scripts/vis_s2.sh hora_v0.0.2
```

4. **Stage-2 headless evaluation**  
   A lighter eval run using `train.algo=ProprioAdapt`, `train.ppo.proprio_adapt=True`, `checkpoint=outputs/AllegroHandHora/hora_v0.0.2/stage2_nn/model_last.ckpt`, and `task.env.numEnvs=2048` showed strong reward growth, confirming the stage-2 pretrained policy is also working correctly.

---

## Next steps (append below)

- Real stage 1 run with default or course-chosen `num_envs` and `minibatch_size` consistency.
- Stage 2: `scripts/train_s2.sh` after stage 1 checkpoints exist.
- Optionally run `vis_s2.sh hora_v0.0.2` or a lighter stage-2 eval now that the pretrained layout is in place.
- Course project experiments (robustness, objects, mjlab, etc.).
