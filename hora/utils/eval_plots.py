from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "rotate_reward",
    "reward",
    "eps_length",
    "lin_vel_x100",
    "command_torque",
]

NUMERIC_FIELDS = {
    "returncode",
    "seed",
    "object_size_longest_edge_m",
    "object_index",
    "object_object_index",
    "progress",
    "max_evaluate_envs",
    "reward",
    "eps_length",
    "rotate_reward",
    "lin_vel_x100",
    "command_torque",
}

BOOLEAN_FIELDS = {
    "use_tactile",
    "use_tactile_obs",
    "use_tactile_hist",
    "use_shape_priv_info",
    "env_use_shape_priv_info",
    "use_extended_priv_info",
}


def resolve_results_csv(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "results.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find results.csv under {path}")
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"Results path does not exist: {path}")
    return path


def _parse_scalar(field: str, value: str) -> Any:
    if value == "":
        return None
    if field in BOOLEAN_FIELDS:
        return value.lower() == "true"
    if field in NUMERIC_FIELDS:
        number = float(value)
        if field in {"returncode", "seed"} and number.is_integer():
            return int(number)
        return number
    return value


def load_results_csv(path: Path) -> list[dict[str, Any]]:
    resolved = resolve_results_csv(path)
    with resolved.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _parse_scalar(key, value) for key, value in row.items()} for row in reader]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _ci95(values: list[float], std_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * std_value / math.sqrt(len(values))


def infer_object_index(object_name: str) -> int | None:
    match = re.match(r"btg(?P<index>\d+)_", object_name)
    if not match:
        return None
    return int(match.group("index"))


def aggregate_results(
    rows: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    metrics = DEFAULT_METRICS if metrics is None else metrics
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault((row["model_name"], row["object_name"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_name, object_name), group_rows in grouped.items():
        base_row = group_rows[0]
        summary: dict[str, Any] = {
            "model_name": model_name,
            "object_name": object_name,
            "object_type": base_row.get("object_type"),
            "use_tactile": base_row.get("use_tactile"),
            "algo": base_row.get("algo"),
            "n_runs": len(group_rows),
            "object_size_longest_edge_m": base_row.get("object_size_longest_edge_m"),
            "object_index": base_row.get("object_index") or infer_object_index(object_name),
        }
        for metric in metrics:
            values = [row[metric] for row in group_rows if row.get(metric) is not None]
            if not values:
                continue
            mean_value = _mean(values)
            std_value = _std(values, mean_value)
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_std"] = std_value
            summary[f"{metric}_ci95"] = _ci95(values, std_value)
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            row.get("object_index") is None,
            row.get("object_index") if row.get("object_index") is not None else 0,
            row.get("object_size_longest_edge_m") is None,
            row.get("object_size_longest_edge_m") if row.get("object_size_longest_edge_m") is not None else row["object_name"],
            row["model_name"],
        )
    )
    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in summary_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def format_summary_markdown(summary_rows: list[dict[str, Any]], metrics: list[str] | None = None) -> str:
    metrics = DEFAULT_METRICS if metrics is None else metrics
    header = [
        "model_name",
        "object_name",
        "size_m",
        "n_runs",
        *[f"{metric}_mean±ci95" for metric in metrics],
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in summary_rows:
        cells = [
            str(row["model_name"]),
            str(row["object_name"]),
            f"{row['object_size_longest_edge_m']:.6f}" if row.get("object_size_longest_edge_m") is not None else "",
            str(row["n_runs"]),
        ]
        for metric in metrics:
            mean_key = f"{metric}_mean"
            ci_key = f"{metric}_ci95"
            if mean_key in row:
                cells.append(f"{row[mean_key]:.4f} ± {row[ci_key]:.4f}")
            else:
                cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary_markdown(summary_rows: list[dict[str, Any]], path: Path, metrics: list[str] | None = None) -> None:
    path.write_text(format_summary_markdown(summary_rows, metrics=metrics))


def plot_metric_curves(
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    metrics: list[str] | None = None,
) -> list[Path]:
    metrics = DEFAULT_METRICS if metrics is None else metrics
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting eval sweep results") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    model_names = sorted({row["model_name"] for row in summary_rows})
    has_numeric_size = all(row.get("object_size_longest_edge_m") is not None for row in summary_rows)

    for metric in metrics:
        mean_key = f"{metric}_mean"
        if not any(mean_key in row for row in summary_rows):
            continue

        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for model_name in model_names:
            model_rows = [row for row in summary_rows if row["model_name"] == model_name and mean_key in row]
            if not model_rows:
                continue
            if has_numeric_size:
                model_rows.sort(key=lambda row: row["object_size_longest_edge_m"])
                x_values = [row["object_size_longest_edge_m"] for row in model_rows]
                axis.set_xlabel("Object longest edge (m)")
            else:
                model_rows.sort(key=lambda row: row["object_name"])
                x_values = list(range(len(model_rows)))
                axis.set_xticks(x_values)
                axis.set_xticklabels([row["object_name"] for row in model_rows], rotation=30, ha="right")
                axis.set_xlabel("Object variant")

            y_values = [row[mean_key] for row in model_rows]
            y_errors = [row.get(f"{metric}_std", 0.0) for row in model_rows]
            axis.errorbar(x_values, y_values, yerr=y_errors, marker="o", linewidth=2, capsize=4, label=model_name)

        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()

        out_path = output_dir / f"{metric}.png"
        figure.savefig(out_path, dpi=160)
        plt.close(figure)
        written_paths.append(out_path)

    return written_paths


def plot_metric_dots(
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    metrics: list[str] | None = None,
) -> list[Path]:
    metrics = DEFAULT_METRICS if metrics is None else metrics
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting eval sweep results") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    model_names = sorted({row["model_name"] for row in summary_rows})

    for metric in metrics:
        mean_key = f"{metric}_mean"
        if not any(mean_key in row for row in summary_rows):
            continue

        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        for model_i, model_name in enumerate(model_names):
            model_rows = [row for row in summary_rows if row["model_name"] == model_name and mean_key in row]
            if not model_rows:
                continue
            model_rows.sort(
                key=lambda row: (
                    row.get("object_index") is None,
                    row.get("object_index") if row.get("object_index") is not None else row["object_name"],
                )
            )
            x_base = [
                row.get("object_index") if row.get("object_index") is not None else i + 1
                for i, row in enumerate(model_rows)
            ]
            offset = (model_i - (len(model_names) - 1) / 2.0) * 0.12
            x_values = [float(x) + offset for x in x_base]
            y_values = [row[mean_key] for row in model_rows]
            y_errors = [row.get(f"{metric}_ci95", 0.0) for row in model_rows]
            axis.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                fmt="o",
                markersize=5,
                capsize=4,
                linestyle="none",
                label=model_name,
            )

        axis.set_title(f"{metric.replace('_', ' ').title()} by BTG Mean Object")
        axis.set_xlabel("Object")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_xticks(range(1, 14))
        axis.set_xticklabels([str(i) for i in range(1, 14)])
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
        figure.tight_layout()

        out_path = output_dir / f"{metric}_dot.png"
        figure.savefig(out_path, dpi=160)
        plt.close(figure)
        written_paths.append(out_path)

    return written_paths


def plot_rotate_reward_seed_scatter(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting eval sweep results") from exc

    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("rotate_reward") is not None]
    if not ok_rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.0, 4.8))

    seeds = sorted({int(row["seed"]) for row in ok_rows if row.get("seed") is not None})
    seed_offsets = {
        seed: ((i - (len(seeds) - 1) / 2.0) * 0.06)
        for i, seed in enumerate(seeds)
    }

    for seed in seeds:
        seed_rows = [row for row in ok_rows if int(row["seed"]) == seed]
        seed_rows.sort(key=lambda row: row.get("object_index") or infer_object_index(row["object_name"]) or row["object_name"])
        x_values = [
            float(row.get("object_index") or infer_object_index(row["object_name"]) or 0) + seed_offsets[seed]
            for row in seed_rows
        ]
        y_values = [row["rotate_reward"] for row in seed_rows]
        axis.scatter(x_values, y_values, s=28, alpha=0.72, label=f"seed {seed}")

    summary_with_rotate = [row for row in summary_rows if row.get("rotate_reward_mean") is not None]
    summary_with_rotate.sort(key=lambda row: row.get("object_index") or infer_object_index(row["object_name"]) or row["object_name"])
    mean_x = [float(row.get("object_index") or infer_object_index(row["object_name"]) or 0) for row in summary_with_rotate]
    mean_y = [row["rotate_reward_mean"] for row in summary_with_rotate]
    mean_err = [row.get("rotate_reward_ci95", 0.0) for row in summary_with_rotate]
    axis.errorbar(
        mean_x,
        mean_y,
        yerr=mean_err,
        fmt="o",
        color="black",
        markersize=5,
        capsize=4,
        linestyle="none",
        label="mean +/- 95% CI",
        zorder=5,
    )

    axis.set_title("Rotation Reward by BTG Mean Object")
    axis.set_xlabel("Object")
    axis.set_ylabel("Rotation Reward")
    axis.set_xticks(range(1, 14))
    axis.set_xticklabels([str(i) for i in range(1, 14)])
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()

    out_path = output_dir / "rotate_reward_seed_scatter.png"
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return [out_path]


def write_eval_summary_outputs(
    results_path: Path,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    metrics = DEFAULT_METRICS if metrics is None else metrics
    results_csv = resolve_results_csv(results_path)
    eval_dir = results_csv.parent
    output_dir = output_dir or (eval_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results_csv(results_csv)
    summary_rows = aggregate_results(rows, metrics=metrics)
    if not summary_rows:
        return [], []

    write_summary_csv(summary_rows, output_dir / "summary.csv")
    write_summary_markdown(summary_rows, output_dir / "summary.md", metrics=metrics)
    plot_paths = plot_rotate_reward_seed_scatter(rows, summary_rows, output_dir)
    return summary_rows, plot_paths
