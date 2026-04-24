#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <manifest.json> [session_name]" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest_input="$1"
session_name="${2:-}"

if [[ "$manifest_input" != /* ]]; then
  manifest_path="$repo_root/$manifest_input"
else
  manifest_path="$manifest_input"
fi

if [[ ! -f "$manifest_path" ]]; then
  echo "Manifest not found: $manifest_path" >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
manifest_stem="$(basename "$manifest_path" .json)"
output_dir="$repo_root/outputs/eval_sweeps/${manifest_stem}_${stamp}"
mkdir -p "$output_dir"

if [[ -z "$session_name" ]]; then
  session_name="eval_${manifest_stem}_${stamp}"
fi

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi

launcher_log="$output_dir/tmux_launcher.log"
cmd="cd '$repo_root' && export TORCH_EXTENSIONS_DIR='$repo_root/.torch_extensions' && . .venv/bin/activate && { . scripts/isaac_wsl_env.sh >/dev/null 2>&1 || true; } && python scripts/eval_object_sweep.py '$manifest_path' --output-dir '$output_dir' 2>&1 | tee '$launcher_log'"

tmux new-session -d -s "$session_name" /bin/bash -lc "$cmd"

echo "Launched detached tmux session: $session_name"
echo "Manifest: $manifest_path"
echo "Output dir: $output_dir"
echo "Launcher log: $launcher_log"
echo
echo "Monitor:"
echo "  tmux attach -t $session_name"
echo "  tail -f '$launcher_log'"
