"""Build a bank from completed repeated PPO runs without spending evaluations again."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data import audit_manifests
from .trajectories import write_bank


def build(root, runs, output):
    root, runs, output = Path(root), Path(runs), Path(output)
    config = json.loads((runs / "plan.json").read_text())["config"]
    inventory = audit_manifests(pd.read_csv(root / config["manifest"]),
                                pd.read_csv(root / config["split_manifest"])).set_index("dataset_key")
    records = []
    for seed in config.get("seeds", [config["seed"]]):
        for key in config["datasets"]:
            if inventory.loc[key, "split"] != "meta_train":
                raise ValueError("Bank input contains a held-out dataset")
            unit = runs / f"{key}-ppo-seed{seed}"
            if not (unit / "completed.json").exists():
                raise ValueError(f"PPO run is incomplete: {unit}")
            episodes = json.loads((unit / "attempt-1" / key / "ppo/episodes.json").read_text())
            if len(episodes) != config["budget"]:
                raise ValueError("Episode count differs from the frozen run budget")
            for record in episodes:
                if record["dataset_key"] != key or record["seed"] != seed:
                    raise ValueError("Trajectory identity differs from run identity")
            records.extend(episodes)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bank(output, records)
    print(f"Saved {len(records)} training-only PPO trajectories", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.runs, args.output)
