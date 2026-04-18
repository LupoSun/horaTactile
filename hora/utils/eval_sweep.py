from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_OVERRIDES = [
    "task=AllegroHandHora",
    "headless=True",
    "pipeline=gpu",
    "test=True",
    "task.on_evaluation=True",
    "task.env.randomization.randomizeMass=False",
    "task.env.randomization.randomizeCOM=False",
    "task.env.randomization.randomizeFriction=False",
    "task.env.randomization.randomizePDGains=False",
    "task.env.randomization.randomizeScale=False",
    "task.env.forceScale=0.0",
    "task.env.randomForceProbScalar=0.0",
]

PROGRESS_PATTERN = re.compile(
    r"progress (?P<progress>\d+) / (?P<max_evaluate_envs>\d+) \| "
    r"reward: (?P<reward>-?\d+(?:\.\d+)?) \| "
    r"eps length: (?P<eps_length>-?\d+(?:\.\d+)?) \| "
    r"rotate reward: (?P<rotate_reward>-?\d+(?:\.\d+)?) \| "
    r"lin vel \(x100\): (?P<lin_vel_x100>-?\d+(?:\.\d+)?) \| "
    r"command torque: (?P<command_torque>-?\d+(?:\.\d+)?)"
)


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "models" not in data or not data["models"]:
        raise ValueError("Manifest must include a non-empty 'models' list")
    if "objects" not in data or not data["objects"]:
        raise ValueError("Manifest must include a non-empty 'objects' list")

    data.setdefault("description", "")
    data.setdefault("num_envs", 4096)
    data.setdefault("max_evaluate_envs", 20000)
    data.setdefault("seeds", [42])
    data.setdefault("base_overrides", [])
    return data


def infer_output_name_from_checkpoint(checkpoint: str) -> str:
    path = Path(checkpoint)
    parts = path.parts
    if "outputs" not in parts:
        raise ValueError(f"Checkpoint path must contain an 'outputs' directory: {checkpoint}")

    outputs_idx = parts.index("outputs")
    try:
        stage_idx = next(i for i in range(outputs_idx + 1, len(parts)) if parts[i] in {"stage1_nn", "stage2_nn"})
    except StopIteration as exc:
        raise ValueError(
            f"Checkpoint path must contain a stage directory like 'stage1_nn' or 'stage2_nn': {checkpoint}"
        ) from exc

    if stage_idx - outputs_idx < 3:
        raise ValueError(f"Checkpoint path is too short to infer train.ppo.output_name: {checkpoint}")

    return "/".join(parts[outputs_idx + 1:stage_idx])


def build_case_name(model: dict[str, Any], obj: dict[str, Any], seed: int) -> str:
    return f"{model['name']}__{obj['name']}__seed{seed}"


def build_eval_command(
    manifest: dict[str, Any],
    model: dict[str, Any],
    obj: dict[str, Any],
    seed: int,
    python_executable: str | None = None,
) -> list[str]:
    algo = model.get("algo", "ProprioAdapt")
    output_name = model.get("output_name") or infer_output_name_from_checkpoint(model["checkpoint"])
    checkpoint = model["checkpoint"]
    use_tactile_obs = bool(model.get("use_tactile_obs", False))
    use_tactile_hist = bool(model.get("use_tactile_hist", model.get("use_tactile", False)))
    priv_info = bool(model.get("priv_info", True))
    proprio_adapt = bool(model.get("proprio_adapt", algo == "ProprioAdapt"))
    python_executable = python_executable or sys.executable

    command = [
        python_executable,
        "train.py",
        *DEFAULT_BASE_OVERRIDES,
        *manifest.get("base_overrides", []),
        f"task.env.numEnvs={manifest['num_envs']}",
        f"task.maxEvaluateEnvs={manifest['max_evaluate_envs']}",
        f"task.env.object.type={obj['object_type']}",
        f"seed={seed}",
        f"train.algo={algo}",
        f"task.env.hora.useTactile={'True' if use_tactile_hist else 'False'}",
        f"task.env.hora.useTactileObs={'True' if use_tactile_obs else 'False'}",
        f"task.env.hora.useTactileHist={'True' if use_tactile_hist else 'False'}",
        f"train.ppo.priv_info={'True' if priv_info else 'False'}",
        f"train.ppo.proprio_adapt={'True' if proprio_adapt else 'False'}",
        f"train.ppo.output_name={output_name}",
        f"checkpoint={checkpoint}",
        *(model.get("extra_overrides", [])),
        *(obj.get("extra_overrides", [])),
    ]
    return command


def parse_eval_metrics(output_text: str) -> dict[str, float] | None:
    matches = list(PROGRESS_PATTERN.finditer(output_text))
    if not matches:
        return None

    match = matches[-1]
    groups = match.groupdict()
    return {
        "progress": float(groups["progress"]),
        "max_evaluate_envs": float(groups["max_evaluate_envs"]),
        "reward": float(groups["reward"]),
        "eps_length": float(groups["eps_length"]),
        "rotate_reward": float(groups["rotate_reward"]),
        "lin_vel_x100": float(groups["lin_vel_x100"]),
        "command_torque": float(groups["command_torque"]),
    }


def default_output_dir(manifest_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "eval_sweeps" / f"{manifest_path.stem}_{stamp}"


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "case_name": result["case_name"],
        "status": result["status"],
        "returncode": result["returncode"],
        "model_name": result["model_name"],
        "algo": result["algo"],
        "use_tactile_obs": result["use_tactile_obs"],
        "use_tactile_hist": result["use_tactile_hist"],
        "object_name": result["object_name"],
        "object_type": result["object_type"],
        "seed": result["seed"],
        "checkpoint": result["checkpoint"],
        "output_name": result["output_name"],
        "log_path": result["log_path"],
    }
    for key, value in result.get("object_metadata", {}).items():
        flat[f"object_{key}"] = value
    for key, value in (result.get("metrics") or {}).items():
        flat[key] = value
    return flat


def write_results_csv(results: list[dict[str, Any]], path: Path) -> None:
    rows = [flatten_result(result) for result in results]
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_results_json(results: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2))


def _stream_subprocess(command: list[str], log_path: Path) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert process.stdout is not None
    with log_path.open("w") as log_handle:
        for line in process.stdout:
            captured.append(line)
            log_handle.write(line)
            print(line, end="")
    returncode = process.wait()
    return returncode, "".join(captured)


def run_sweep(
    manifest_path: Path,
    output_dir: Path | None = None,
    python_executable: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    output_dir = default_output_dir(manifest_path) if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    (output_dir / "manifest.snapshot.json").write_text(json.dumps(manifest, indent=2))

    results: list[dict[str, Any]] = []
    for model in manifest["models"]:
        for obj in manifest["objects"]:
            for seed in manifest["seeds"]:
                case_name = build_case_name(model, obj, seed)
                command = build_eval_command(manifest, model, obj, seed, python_executable=python_executable)
                log_path = logs_dir / f"{case_name}.log"

                result = {
                    "case_name": case_name,
                    "status": "dry_run" if dry_run else "pending",
                    "returncode": None,
                    "model_name": model["name"],
                    "algo": model.get("algo", "ProprioAdapt"),
                    "use_tactile_obs": bool(model.get("use_tactile_obs", False)),
                    "use_tactile_hist": bool(model.get("use_tactile_hist", model.get("use_tactile", False))),
                    "object_name": obj["name"],
                    "object_type": obj["object_type"],
                    "object_metadata": {
                        key: value for key, value in obj.items() if key not in {"name", "object_type", "extra_overrides"}
                    },
                    "seed": seed,
                    "checkpoint": model["checkpoint"],
                    "output_name": model.get("output_name") or infer_output_name_from_checkpoint(model["checkpoint"]),
                    "log_path": str(log_path.relative_to(output_dir)),
                    "command": command,
                    "metrics": None,
                }

                if dry_run:
                    results.append(result)
                    continue

                print(f"\n=== Running {case_name} ===")
                returncode, output_text = _stream_subprocess(command, log_path)
                result["returncode"] = returncode
                result["metrics"] = parse_eval_metrics(output_text)
                result["status"] = "ok" if returncode == 0 and result["metrics"] is not None else "error"
                results.append(result)

                _write_results_json(results, output_dir / "results.json")
                write_results_csv(results, output_dir / "results.csv")

    if not dry_run:
        _write_results_json(results, output_dir / "results.json")
        write_results_csv(results, output_dir / "results.csv")

    return results
