#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mesh.preprocess import process_mesh


BENCHMARK_MEAN_LONGEST_EDGE_M = 0.0871111111111111
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
DEFAULT_VARIANTS = [
    ("original", 0.0),
    ("lerp_25", 0.25),
    ("lerp_50", 0.50),
    ("lerp_75", 0.75),
    ("mean", 1.0),
]

DEFAULT_MODELS = [
    {
        "name": "vanilla_stage2",
        "checkpoint": "outputs/AllegroHandHora/hora_v0.0.2/stage2_nn/model_best.ckpt",
        "algo": "ProprioAdapt",
        "use_tactile_obs": False,
        "use_tactile_hist": False,
    },
    {
        "name": "tactile_stage1_release",
        "checkpoint": "outputs/AllegroHandHora/double_tactile_s1_2/stage1_nn/best.pth",
        "algo": "PPO",
        "use_tactile_obs": True,
        "use_tactile_hist": False,
        "priv_info": True,
        "proprio_adapt": False,
    },
    {
        "name": "tactile_stage2_release",
        "checkpoint": "outputs/AllegroHandHora/double_tactile_s1_2/stage2_nn/model_best.ckpt",
        "algo": "ProprioAdapt",
        "use_tactile_obs": True,
        "use_tactile_hist": True,
        "priv_info": True,
        "proprio_adapt": True,
    },
]

DEFAULT_BASE_OVERRIDES = [
    "task.env.reset_height_threshold=0.6",
    "task.env.randomization.jointNoiseScale=0.0",
    "task.env.baseObjScale=1.0",
    "task.env.randomization.graspInitScale=0.8",
]


def btg_prefix(mesh_stem: str) -> str:
    digits = "".join(re.findall(r"\d+", mesh_stem))
    if not digits:
        raise ValueError(f"Could not extract BTG index from mesh stem '{mesh_stem}'")
    return f"btg{int(digits)}"


def load_metadata(obj_dir: Path) -> dict:
    metadata_path = obj_dir / "metadata.json"
    return json.loads(metadata_path.read_text())


def ensure_variant(
    mesh_path: Path,
    subset_dir: Path,
    export_unit_scale: float,
    target_bbox_size: float | None,
    force: bool,
) -> dict:
    obj_dir = subset_dir / mesh_path.stem
    metadata_path = obj_dir / "metadata.json"

    if force or not metadata_path.exists():
        process_mesh(
            mesh_path=mesh_path,
            output_dir=subset_dir,
            point_counts=[100, 1024],
            export_unit_scale=export_unit_scale,
            target_bbox_size=target_bbox_size,
        )

    return load_metadata(obj_dir)


def build_manifest(
    objects: list[dict],
    description: str,
    num_envs: int,
    max_evaluate_envs: int,
) -> dict:
    return {
        "description": description,
        "num_envs": num_envs,
        "max_evaluate_envs": max_evaluate_envs,
        "seeds": DEFAULT_SEEDS,
        "base_overrides": DEFAULT_BASE_OVERRIDES,
        "models": DEFAULT_MODELS,
        "objects": objects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5-size BTG asset ladders for remaining Nodes/*.stl meshes and emit a sweep manifest.",
    )
    parser.add_argument(
        "--nodes-dir",
        type=Path,
        default=REPO_ROOT / "Nodes",
        help="Directory containing BTG_*.stl meshes.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=REPO_ROOT / "assets" / "custom",
        help="Root assets/custom directory for generated object bundles.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPO_ROOT / "configs" / "eval_sweeps" / "btg_nodes_rest_vanilla_stage1_stage2_release_10seeds.json",
        help="Path to write the sweep manifest JSON.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["BTG_13"],
        help="Mesh stems to exclude from the generated sweep.",
    )
    parser.add_argument(
        "--mean-longest-edge",
        type=float,
        default=BENCHMARK_MEAN_LONGEST_EDGE_M,
        help="Target benchmark mean longest-edge size in meters.",
    )
    parser.add_argument(
        "--export-unit-scale",
        type=float,
        default=0.001,
        help="Unit conversion applied to raw STL meshes before bbox sizing.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Parallel env count for the generated evaluation manifest.",
    )
    parser.add_argument(
        "--max-evaluate-envs",
        type=int,
        default=5000,
        help="Evaluation episode count per case for the generated evaluation manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild asset bundles even if metadata.json already exists.",
    )
    args = parser.parse_args()

    mesh_paths = sorted(args.nodes_dir.glob("BTG_*.stl"))
    exclude = set(args.exclude)
    mesh_paths = [path for path in mesh_paths if path.stem not in exclude]
    if not mesh_paths:
        raise SystemExit("No BTG meshes matched after applying --exclude.")

    objects: list[dict] = []
    for mesh_path in mesh_paths:
        prefix = btg_prefix(mesh_path.stem)

        original_subset = args.assets_root / f"{prefix}_original"
        original_meta = ensure_variant(
            mesh_path=mesh_path,
            subset_dir=original_subset,
            export_unit_scale=args.export_unit_scale,
            target_bbox_size=None,
            force=args.force,
        )
        original_longest = max(original_meta["export_bbox"])

        for variant_name, lerp_t in DEFAULT_VARIANTS:
            subset_dir = args.assets_root / f"{prefix}_{variant_name}"
            target_size = None
            if variant_name != "original":
                target_size = original_longest + lerp_t * (args.mean_longest_edge - original_longest)

            meta = ensure_variant(
                mesh_path=mesh_path,
                subset_dir=subset_dir,
                export_unit_scale=args.export_unit_scale,
                target_bbox_size=target_size,
                force=args.force,
            )

            objects.append(
                {
                    "name": f"{prefix}_{variant_name}",
                    "object_type": f"custom_{prefix}_{variant_name}",
                    "size_longest_edge_m": max(meta["export_bbox"]),
                    "source_mesh": mesh_path.name,
                    "size_variant": variant_name,
                    "lerp_t": lerp_t,
                }
            )

    description = (
        "Compare vanilla HORA Stage 2, tactile Stage 1, and tactile Stage 2 on the "
        "5-size ladders for the remaining BTG meshes in Nodes/."
    )
    manifest = build_manifest(
        objects=objects,
        description=description,
        num_envs=args.num_envs,
        max_evaluate_envs=args.max_evaluate_envs,
    )

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2))

    num_cases = len(manifest["models"]) * len(manifest["objects"]) * len(manifest["seeds"])
    print(f"Prepared {len(objects)} object variants from {len(mesh_paths)} meshes.")
    print(f"Manifest: {args.manifest_output}")
    print(f"Total sweep cases: {num_cases}")


if __name__ == "__main__":
    main()
