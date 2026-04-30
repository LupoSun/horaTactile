from __future__ import annotations

import csv
import json
import os
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

CASE_BAR_WIDTH = 24


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
    use_shape_priv_info = bool(model.get("use_shape_priv_info", False))
    env_use_shape_priv_info = bool(model.get("env_use_shape_priv_info", use_shape_priv_info))
    use_extended_priv_info = bool(model.get("use_extended_priv_info", False))
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
        f"task.env.hora.useTactileObs={'True' if use_tactile_obs else 'False'}",
        f"task.env.hora.useTactileHist={'True' if use_tactile_hist else 'False'}",
        f"task.env.hora.useShapePrivInfo={'True' if env_use_shape_priv_info else 'False'}",
        f"task.env.hora.useExtendedPrivInfo={'True' if use_extended_priv_info else 'False'}",
        f"train.ppo.use_shape_priv_info={'True' if use_shape_priv_info else 'False'}",
        f"task.env.hora.privInfoDim={model.get('priv_info_dim', 17 if use_extended_priv_info else 9)}",
        f"train.ppo.priv_info_dim={model.get('priv_info_dim', 17 if use_extended_priv_info else 9)}",
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
        "use_shape_priv_info": result["use_shape_priv_info"],
        "env_use_shape_priv_info": result["env_use_shape_priv_info"],
        "use_extended_priv_info": result["use_extended_priv_info"],
        "object_name": result["object_name"],
        "object_type": result["object_type"],
        "seed": result["seed"],
        "checkpoint": result["checkpoint"],
        "output_name": result["output_name"],
        "log_path": result["log_path"],
    }
    for key, value in result.get("object_metadata", {}).items():
        flat[f"object_{key}"] = value
        if key == "object_index":
            flat["object_index"] = value
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


def _progress_bar(done: int, total: int, width: int = CASE_BAR_WIDTH) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = min(width, int(width * done / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _print_case_progress(
    completed: int,
    total: int,
    case_name: str,
    status: str = "running",
    detail: str = "",
    end: str = "\r",
) -> None:
    line = f"{_progress_bar(completed, total)} {completed}/{total} {status}: {case_name}"
    if detail:
        line = f"{line} | {detail}"
    print(line[:220].ljust(220), end=end, flush=True)


def _last_nonempty_lines(output_text: str, limit: int = 8) -> list[str]:
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    return lines[-limit:]


def eval_case_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["WANDB_MODE"] = "disabled"
    return env


def init_sweep_wandb_run(
    output_dir: Path,
    run_name: str,
    group: str = "eval",
    project: str = "hora",
):
    if not run_name:
        return None

    try:
        import wandb
        from hora.utils.wandb_utils import resolve_wandb_entity, resolve_wandb_mode
    except ImportError as exc:
        raise RuntimeError("wandb is required to log eval sweep results") from exc

    mode = resolve_wandb_mode()
    if mode == "disabled":
        return None
    if wandb.run is not None:
        return wandb.run

    run = wandb.init(
        entity=resolve_wandb_entity(),
        project=project,
        name=run_name,
        group=group,
        mode=mode,
        config={
            "eval_output_dir": str(output_dir),
            "status": "running",
        },
    )
    if getattr(run, "url", None):
        print(f"[hora] W&B eval run: {run.url}")
    return run


def log_eval_case_to_wandb(result: dict[str, Any], completed: int, total: int) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    payload: dict[str, Any] = {
        "eval/cases_completed": completed,
        "eval/cases_total": total,
        "eval/cases_fraction": completed / total if total else 0.0,
    }

    wandb.log(payload, step=completed)


def _stream_subprocess(
    command: list[str],
    log_path: Path,
    case_name: str,
    completed: int,
    total: int,
) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=eval_case_subprocess_env(),
    )
    captured: list[str] = []
    assert process.stdout is not None
    with log_path.open("w") as log_handle:
        for line in process.stdout:
            captured.append(line)
            log_handle.write(line)
            match = PROGRESS_PATTERN.search(line)
            if match:
                groups = match.groupdict()
                detail = (
                    f"eval {groups['progress']}/{groups['max_evaluate_envs']} "
                    f"reward={groups['reward']} rotate={groups['rotate_reward']}"
                )
                _print_case_progress(completed, total, case_name, detail=detail)
    returncode = process.wait()
    return returncode, "".join(captured)


def run_sweep(
    manifest_path: Path,
    output_dir: Path | None = None,
    python_executable: str | None = None,
    dry_run: bool = False,
    wandb_name: str = "",
    wandb_group: str = "eval",
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    output_dir = default_output_dir(manifest_path) if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    (output_dir / "manifest.snapshot.json").write_text(json.dumps(manifest, indent=2))
    if not dry_run and wandb_name:
        init_sweep_wandb_run(output_dir, wandb_name, group=wandb_group)

    results: list[dict[str, Any]] = []
    total_cases = len(manifest["models"]) * len(manifest["objects"]) * len(manifest["seeds"])
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
                    "use_shape_priv_info": bool(model.get("use_shape_priv_info", False)),
                    "env_use_shape_priv_info": bool(model.get("env_use_shape_priv_info", model.get("use_shape_priv_info", False))),
                    "use_extended_priv_info": bool(model.get("use_extended_priv_info", False)),
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

                _print_case_progress(len(results), total_cases, case_name)
                returncode, output_text = _stream_subprocess(
                    command,
                    log_path,
                    case_name=case_name,
                    completed=len(results),
                    total=total_cases,
                )
                result["returncode"] = returncode
                result["metrics"] = parse_eval_metrics(output_text)
                result["status"] = "ok" if returncode == 0 and result["metrics"] is not None else "error"
                results.append(result)

                if result["status"] == "ok":
                    metrics = result["metrics"] or {}
                    detail = f"reward={metrics.get('reward', 0):.2f} rotate={metrics.get('rotate_reward', 0):.2f}"
                    _print_case_progress(len(results), total_cases, case_name, status="ok", detail=detail, end="\n")
                else:
                    _print_case_progress(
                        len(results),
                        total_cases,
                        case_name,
                        status="error",
                        detail=f"see {log_path.relative_to(output_dir)}",
                        end="\n",
                    )
                    for line in _last_nonempty_lines(output_text):
                        print(f"  {line}")

                _write_results_json(results, output_dir / "results.json")
                write_results_csv(results, output_dir / "results.csv")
                log_eval_case_to_wandb(result, completed=len(results), total=total_cases)

    if not dry_run:
        _write_results_json(results, output_dir / "results.json")
        write_results_csv(results, output_dir / "results.csv")

    return results


def log_sweep_to_wandb(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    plot_paths: list[Path],
    run_name: str,
    group: str = "eval",
    project: str = "hora",
) -> None:
    if not summary_rows:
        return

    try:
        import wandb
        from hora.utils.wandb_utils import resolve_wandb_entity, resolve_wandb_mode
    except ImportError as exc:
        raise RuntimeError("wandb is required to log eval sweep results") from exc

    mode = resolve_wandb_mode()
    if mode == "disabled":
        return

    run = wandb.run
    if run is None:
        run = wandb.init(
            entity=resolve_wandb_entity(),
            project=project,
            name=run_name,
            group=group,
            mode=mode,
            config={
                "eval_output_dir": str(output_dir),
                "n_summary_rows": len(summary_rows),
                "status": "complete",
            },
        )
    elif hasattr(run, "config"):
        run.config.update({"n_summary_rows": len(summary_rows), "status": "complete"}, allow_val_change=True)

    columns = sorted({key for row in summary_rows for key in row.keys()})
    table = wandb.Table(columns=columns)
    for row in summary_rows:
        table.add_data(*[row.get(column) for column in columns])

    log_payload: dict[str, Any] = {"eval/summary": table}
    for plot_path in plot_paths:
        if plot_path.is_file():
            log_payload[f"eval/{plot_path.stem}"] = wandb.Image(str(plot_path))
    wandb.log(log_payload)

    for row in summary_rows:
        object_index = row.get("object_index")
        step = int(object_index) if object_index is not None else None
        scalar_payload = {
            f"eval/{row['object_name']}/{key}": value
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"object_index"}
        }
        if scalar_payload:
            wandb.log(scalar_payload, step=step)

    artifact = wandb.Artifact(f"{run_name.replace('/', '_')}_outputs", type="eval_sweep")
    for relpath in ("results.csv", "results.json", "manifest.snapshot.json"):
        path = output_dir / relpath
        if path.is_file():
            artifact.add_file(str(path), name=relpath)
    plots_dir = output_dir / "plots"
    if plots_dir.is_dir():
        artifact.add_dir(str(plots_dir), name="plots")
    run.log_artifact(artifact)
    wandb.finish()


def finalize_sweep_outputs(
    output_dir: Path,
    wandb_name: str = "",
    wandb_group: str = "eval",
) -> tuple[list[dict[str, Any]], list[Path]]:
    from hora.utils.eval_plots import write_eval_summary_outputs

    summary_rows, plot_paths = write_eval_summary_outputs(output_dir)
    if wandb_name:
        log_sweep_to_wandb(output_dir, summary_rows, plot_paths, run_name=wandb_name, group=wandb_group)
    return summary_rows, plot_paths
