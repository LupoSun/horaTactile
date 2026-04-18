from pathlib import Path

from hora.utils.eval_plots import (
    aggregate_results,
    format_summary_markdown,
    load_results_csv,
    resolve_results_csv,
    write_summary_csv,
)


def test_resolve_results_csv_accepts_directory(tmp_path):
    results_csv = tmp_path / "results.csv"
    results_csv.write_text("status,model_name,object_name\n")
    assert resolve_results_csv(tmp_path) == results_csv


def test_load_results_csv_parses_numeric_fields(tmp_path):
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "status,model_name,object_name,use_tactile,seed,rotate_reward,object_size_longest_edge_m\n"
        "ok,baseline,mean,False,42,1.25,0.1\n"
    )
    rows = load_results_csv(results_csv)
    assert rows[0]["use_tactile"] is False
    assert rows[0]["seed"] == 42
    assert rows[0]["rotate_reward"] == 1.25
    assert rows[0]["object_size_longest_edge_m"] == 0.1


def test_aggregate_results_computes_mean_and_std():
    rows = [
        {
            "status": "ok",
            "model_name": "baseline",
            "object_name": "mean",
            "object_type": "custom_btg13_mean",
            "use_tactile": False,
            "algo": "ProprioAdapt",
            "seed": 42,
            "rotate_reward": 1.0,
            "reward": 2.0,
            "eps_length": 100.0,
            "lin_vel_x100": 0.1,
            "command_torque": 0.2,
            "object_size_longest_edge_m": 0.1,
        },
        {
            "status": "ok",
            "model_name": "baseline",
            "object_name": "mean",
            "object_type": "custom_btg13_mean",
            "use_tactile": False,
            "algo": "ProprioAdapt",
            "seed": 43,
            "rotate_reward": 3.0,
            "reward": 4.0,
            "eps_length": 200.0,
            "lin_vel_x100": 0.3,
            "command_torque": 0.6,
            "object_size_longest_edge_m": 0.1,
        },
    ]
    summary_rows = aggregate_results(rows, metrics=["rotate_reward", "reward"])
    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row["n_runs"] == 2
    assert row["rotate_reward_mean"] == 2.0
    assert round(row["rotate_reward_std"], 6) == round(2 ** 0.5, 6)
    assert row["reward_mean"] == 3.0


def test_summary_markdown_includes_metric_columns():
    summary_rows = [
        {
            "model_name": "baseline",
            "object_name": "mean",
            "object_size_longest_edge_m": 0.1,
            "n_runs": 3,
            "rotate_reward_mean": 1.23,
            "rotate_reward_std": 0.45,
        }
    ]
    markdown = format_summary_markdown(summary_rows, metrics=["rotate_reward"])
    assert "rotate_reward_mean±std" in markdown
    assert "1.2300 ± 0.4500" in markdown


def test_write_summary_csv_writes_dynamic_headers(tmp_path):
    path = tmp_path / "summary.csv"
    write_summary_csv(
        [
            {
                "model_name": "baseline",
                "object_name": "mean",
                "rotate_reward_mean": 1.5,
            }
        ],
        path,
    )
    text = path.read_text()
    assert "rotate_reward_mean" in text
    assert "baseline" in text
