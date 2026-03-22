#!/usr/bin/env bash
# Safer defaults for WSLg: CPU PhysX, cartpole, and num_envs=1 (upstream joint_monkey uses 36).
# Usage: bash scripts/wsl_joint_monkey.sh
# Extra args are forwarded, e.g. bash scripts/wsl_joint_monkey.sh --asset_id 0 --num_envs 4
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/isaac_wsl_env.sh"
cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" "${ROOT}/scripts/joint_monkey_wsl.py" --sim_device cpu --asset_id 2 --num_envs 1 "$@"
