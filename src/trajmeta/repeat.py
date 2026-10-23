"""Sequential repeated-seed pilot with hash-checked completed-run resume."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import yaml

from .cli import load_config, run


def execute(config_path, output, root=None):
    config_path, output = Path(config_path), Path(output)
    root = Path(root or config_path.resolve().parent.parent)
    config = load_config(config_path)
    seeds = config.get("seeds", [config["seed"]])
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    identity = sha256(config_path.read_bytes())
    for path in [root / config["manifest"], root / config["split_manifest"], *sorted((root / "src/trajmeta").glob("*.py"))]:
        identity.update(path.name.encode())
        identity.update(path.read_bytes())
    signature = identity.hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    plan = output / "plan.json"
    if plan.exists():
        if json.loads(plan.read_text())["signature"] != signature:
            raise ValueError("Cannot resume with changed code, config, or manifests")
    else:
        plan.write_text(json.dumps({"signature": signature, "seeds": seeds, "config": config}, indent=2))
    summaries = []
    for seed in seeds:
        for dataset in config["datasets"]:
            for method in config["methods"]:
                unit = output / f"{dataset}-{method}-seed{seed}"
                unit.mkdir(exist_ok=True)
                completed = unit / "completed.json"
                if completed.exists():
                    summaries.extend(json.loads(completed.read_text()))
                    print(f"resume: skipping completed {unit.name}", flush=True)
                    continue
                attempts = list(unit.glob("attempt-*"))
                if attempts:
                    raise RuntimeError(f"Interrupted run at {unit}; inspect partial evaluation logs and use a new output folder. Automatic retries could hide compute cost.")
                unit_config = {**config, "seed": seed, "datasets": [dataset], "methods": [method]}
                path = unit / "config.yaml"
                path.write_text(yaml.safe_dump(unit_config))
                result = run(path, unit / "attempt-1", root=root)
                pending = unit / "completed.tmp"
                pending.write_text(json.dumps(result, indent=2))
                pending.replace(completed)
                summaries.extend(result)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    execute(args.config, args.output)
