"""JSON-only records and strict bank split admission."""
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1


def validate_trajectory(record, train_only=True):
    required = {"schema_version", "trajectory_id", "dataset_key", "split", "seed", "policy_version",
                "steps", "pipeline_config", "inner_score", "elapsed_seconds", "status", "provenance_hash"}
    if not required.issubset(record) or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported or incomplete trajectory")
    if train_only and record["split"] != "meta_train":
        raise ValueError("Only meta-training trajectories can enter the bank")
    steps = record["steps"]
    if len(steps) != 4 or [s["done"] for s in steps] != [False, False, False, True]:
        raise ValueError("Trajectory must contain one complete four-step episode")
    width = len(steps[0]["state"])
    for i, step in enumerate(steps):
        if len(step["state"]) != width or len(step["next_state"]) != width:
            raise ValueError("Inconsistent state schema")
        if not all(math.isfinite(v) for v in step["state"] + step["next_state"] + [step["base_reward"]]):
            raise ValueError("Non-finite trajectory")
        if i and steps[i - 1]["next_state"] != step["state"]:
            raise ValueError("Discontinuous state sequence")
    if steps[-1]["base_reward"] != record["inner_score"]:
        raise ValueError("Terminal reward and evaluation score differ")


def write_bank(path, records):
    seen = set()
    for record in records:
        validate_trajectory(record)
        if record["trajectory_id"] in seen:
            raise ValueError("Duplicate trajectory ID")
        seen.add(record["trajectory_id"])
    Path(path).write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")


def read_bank(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        validate_trajectory(record)
    if len({r["trajectory_id"] for r in records}) != len(records):
        raise ValueError("Duplicate trajectory ID")
    return records
